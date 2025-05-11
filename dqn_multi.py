# encoding: utf8
import random
import pickle
import numpy as np
from collections import deque
from tensorflow import keras, constant
from keras.models import Sequential, Model
from keras.layers import *
from turtlesim_env_base import TurtlesimEnvBase
import turtlesim_env_multi
from dqn_single import DqnSingle


class DqnMulti(DqnSingle):
    def __init__(self, env: TurtlesimEnvBase, id_prefix="dqnm", seed=42):
        super().__init__(env, id_prefix, seed)

    # złożenie dwóch rastrów sytuacji aktualnej i poprzedniej w tensor 5x5x10 wejścia do sieci
    def inp_stack(_, last, cur):
        # fa,fd,fc+1,fp+1 ORAZ fo doklejone na końcu
        inp = np.stack([cur[2], cur[3], cur[4], cur[5], last[2], last[3], last[4], last[5], cur[6], last[6]], axis=-1)
        return inp

    # predykcja nagród łącznych (Q) za sterowania na podst. bieżącej i ostatniej sytuacji
    # wytworzenie modelu - sieci neuronowej
    def make_model(self):
        N = self.env.GRID_RES  # rozdzielczość rastra
        M = 10  # liczba warstw z inp_stack()
        self.model = Sequential()
        self.model.add(Conv3D(filters=2 * M, kernel_size=(2, 2, M), activation="relu", input_shape=(N, N, M, 1)))
        self.model.add(Permute((1, 2, 4, 3)))
        self.model.add(Conv3D(filters=2 * M, kernel_size=(2, 2, 2 * M), activation="relu"))
        self.model.add(Permute((1, 2, 4, 3)))
        self.model.add(Conv3D(filters=2 * M, kernel_size=(2, 2, 2 * M), activation="relu"))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation="relu"))  # (128)
        self.model.add(Dense(self.CTL_DIM, activation="linear"))  # wyjście Q dla każdej z CTL_DIM decyzji
        self.model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])

    def make_branched_model(self):
        N = self.env.GRID_RES  # rozdzielczość rastra
        # gałąź analizy trasy
        M = 8
        input = Input(
            shape=(N, N, M, 1)
        )  # bezpośrednie użycie input_shape w conv3d daje błąd przy podłączaniu kolejnych warstw
        conv3d = Conv3D(filters=2 * M, kernel_size=(2, 2, M), activation="relu", name="conv3d")(input)
        permute = Permute((1, 2, 4, 3))(conv3d)
        conv3d_1 = Conv3D(filters=2 * M, kernel_size=(2, 2, 2 * M), activation="relu", name="conv3d_1")(permute)
        permute_1 = Permute((1, 2, 4, 3))(conv3d_1)
        conv3d_2 = Conv3D(filters=2 * M, kernel_size=(2, 2, 2 * M), activation="relu", name="conv3d_2")(permute_1)
        flatten = Flatten()(conv3d_2)
        # gałąź analizy kolizji
        M = 2
        inputX = Input(shape=(N, N, M, 1))
        conv3dX = Conv3D(filters=2 * M, kernel_size=(2, 2, M), activation="relu", name="conv3dX")(inputX)
        permuteX = Permute((1, 2, 4, 3))(conv3dX)
        conv3dX_1 = Conv3D(filters=2 * M, kernel_size=(2, 2, 2 * M), activation="relu", name="conv3dX_1")(permuteX)
        permuteX_1 = Permute((1, 2, 4, 3))(conv3dX_1)
        conv3dX_2 = Conv3D(filters=2 * M, kernel_size=(2, 2, 2 * M), activation="relu", name="conv3dX_2")(permuteX_1)
        flattenX = Flatten()(conv3dX_2)
        # połączenie gałęzi i gęsta końcówka
        combined = Concatenate()([flatten, flattenX])
        dense = Dense(32, activation="relu")(combined)
        dense_1 = Dense(6, activation="linear")(dense)
        self.model = Model([input, inputX], dense_1)
        self.model.get_layer("conv3dX_2").trainable = False
        self.model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])

    # model z osobną gałęzią dla logiki unikania kolizji
    def train_main(self, save_model=True, save_state=True):
        self.target_model = keras.models.clone_model(self.model)  # model pomocniczy (wolnozmienny)
        self.replay_memory = deque(maxlen=self.REPLAY_MEM_SIZE_MAX)  # historia kroków
        episode_rewards = np.zeros(self.EPISODES_MAX) * np.nan  # historia nagród w epizodach
        epsilon = self.EPS_INIT
        step_cnt = 0
        train_cnt = 0
        current_states = {tname: agent.map for tname, agent in self.env.agents.items()}  # aktualne sytuacje
        last_states = {
            tname: agent.map for tname, agent in self.env.agents.items()
        }  # poprzednie stytuacje początkowo takie same
        agent_episode = {tname: i for i, tname in enumerate(self.env.agents)}  # indeks epizodu przypisany do agenta
        episode_rewards[: len(self.env.agents)] = 0  # inicjalizacja nagród za epizody
        episode = len(self.env.agents) - 1  # indeks ost. epizodu
        to_restart = set()  # agenty do reaktywacji
        while episode < self.EPISODES_MAX:  # ucz w epizodach treningowych
            self.env.reset(to_restart, ["random" for i in to_restart])  # inicjalizacja wybranych
            for tname in to_restart:  # odczytanie sytuacji
                current_states[tname] = self.env.agents[tname].map  # początkowa sytuacja
                last_states[tname] = [
                    i.copy() for i in current_states[tname]
                ]  # zaczyna od postoju: poprz. stan taki jak obecny
                episode += 1  # dla niego to nowy epizod
                episode_rewards[episode] = 0  # inicjalizacja nagród w tym epizodzie
                agent_episode[tname] = episode  # przypisanie agenta do epizodu
                if (episode + 1) % self.SAVE_MODEL_EVERY == 0 and save_model:
                    self.model.save(f"models/{self.xid()}-{episode+1}.tf")  # zapisz bieżący model na dysku
                if (episode + 1) % self.SAVE_MODEL_EVERY == 0 and save_state:  # zapisz bieżący stan uczenia
                    pickle.dump(
                        (episode, episode_rewards, epsilon, self.replay_memory), open(f"models/{self.xid()}.pkl", "wb")
                    )
            to_restart = set()
            controls = {}  # sterowania poszczególnych agentów
            for tname in self.env.agents:  # poruszamy każdym agentem
                if np.random.random() > epsilon:  # sterowanie wg reguły albo losowe
                    controls[tname] = np.argmax(self.decision(self.model, last_states[tname], current_states[tname]))
                    print("o", end="")
                else:
                    controls[tname] = np.random.randint(0, self.CTL_DIM)  # losowa prędkość pocz. i skręt
                    print(".", end="")
            actions = {tname: self.ctl2act(control) for tname, control in controls.items()}  # wartości sterowań
            scene = self.env.step(actions)  # kroki i wyniki symulacji
            for tname, (new_state, reward, done) in scene.items():  # obsługa po kroku dla każdego agenta
                episode_rewards[agent_episode[tname]] += reward  # akumulacja nagrody
                self.replay_memory.append(
                    (last_states[tname], current_states[tname], controls[tname], reward, new_state, done)
                )
                step_cnt += 1
                if len(self.replay_memory) >= self.REPLAY_MEM_SIZE_MIN and step_cnt % self.TRAIN_EVERY == 0:
                    self.do_train(episode=episode)  # ucz, gdy zgromadzono dość próbek
                    train_cnt += 1
                    if train_cnt % self.UPDATE_TARGET_EVERY == 0:
                        self.target_model.set_weights(self.model.get_weights())  # aktualizuj model pomocniczy
                        print("T", end="")
                    else:
                        print("t", end="")
                if done:
                    to_restart.add(tname)
                    print(f"\n {len(self.replay_memory)} {tname} E{episode} ", end="")
                    print(
                        f'{np.nanmean(episode_rewards.take(range(episode-self.env.MAX_STEPS-1,episode+1),mode="wrap"))/self.env.MAX_STEPS:.2f} ',
                        end="",
                    )  # śr. nagroda za krok
                last_states[tname] = current_states[tname]  # przejście do nowego stanu
                current_states[tname] = new_state  # z zapamiętaniem poprzedniego
                if epsilon > self.EPS_MIN:  # rosnące p-stwo uczenia na podst. historii
                    epsilon *= self.EPS_DECAY
                    epsilon = max(self.EPS_MIN, epsilon)  # ogranicz malenie eps


# przykładowe wywołanie uczenia
from tensorflow.keras.models import load_model

if __name__ == "__main__":
    env = turtlesim_env_multi.provide_env()  # utworzenie środowiska
    env.PI_BY = 3  # zmiana wybranych parametrów środowiska
    prefix = "X6-c20c20c20d64-M-lr001"  # bazowy z kolizjami
    env.DETECT_COLLISION = True
    env.setup("routes.csv")  # połączenie z symulatorem
    agents = env.reset()  # ustawienie agenta
    dqnm = DqnMulti(env, id_prefix=prefix)  # utworzenie klasy uczącej
    dqnm.make_model()  # skonstruowanie sieci neuronowej
    dqnm.train_main(save_model=True, save_state=True)  # wywołanie uczenia (wyniki zapisywane okresowo)

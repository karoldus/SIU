import sys
import numpy as np
from tensorflow import keras
from copy import deepcopy
import turtlesim
from turtlesim.srv import SetPenRequest
from turtlesim_env_single import TurtlesimEnvSingle
from turtlesim_env_multi import TurtlesimEnvMulti


class AppSingle:
    def __init__(self, model: str):
        self.model = keras.models.load_model(model)
        self.set_pen = False

    # zakodowanie wybranego sterowania (0-5) na potrzeby środowiska: (prędkość,skręt)
    def ctl2act(_, decision: int):  # prędkość\skręt    -.1rad 0 .1rad
        v = 0.2  #   0.2                0   1   2
        if decision >= 3:  #   0.4                3   4   5
            v = 0.4
        w = 0.25 * (decision % 3 - 1)
        return [v, w]

    # złożenie dwóch rastrów sytuacji aktualnej i poprzedniej w tensor 5x5x8 wejścia do sieci
    def inp_stack(_, last, cur):
        # fa,fd,fc+1,fp+1 - z wyjścia get_map - BEZ 2 POCZ. WARTOŚCI (zalecana prędkość w ukł. odniesienia planszy)
        inp = np.stack([cur[2], cur[3], cur[4], cur[5], last[2], last[3], last[4], last[5]], axis=-1)
        return inp

    # predykcja nagród łącznych (Q) za sterowania na podst. bieżącej i ostatniej sytuacji
    def decision(self, the_model, last, cur):
        inp = np.expand_dims(self.inp_stack(last, cur), axis=-1)
        inp = np.expand_dims(inp, axis=0)
        # return the_model.predict(inp,verbose=0).flatten() # wektor przewidywanych nagród dla sterowań -> UBYTEK PAMIĘCI w dockerze
        return the_model(inp).numpy().flatten()  # wektor przewidywanych nagród dla sterowań

    def app(self):
        env = TurtlesimEnvSingle()
        env.setup("routes.csv", agent_cnt=1)
        agents = env.reset()
        tname = list(agents.keys())[0]
        # set pen
        # set_pen_req = turtlesim.srv.SetPenRequest(r=255, g=0, b=255, width=5, off=0)
        # env.tapi.setPen(tname,set_pen_req)
        current_state = deepcopy(agents[tname].map)
        while not env.out_of_track:
            last_state = deepcopy(current_state)
            control = np.argmax(self.decision(self.model, last_state, current_state))
            current_state, _, _ = env.step({tname: self.ctl2act(control)}, realtime=False)


class AppMulti:
    def __init__(self, model: str, agent_cnt=5):
        self.model = keras.models.load_model(model)
        self.agent_cnt = agent_cnt

    def ctl2act(self, decision: int):
        v = 0.2
        if decision >= 3:
            v = 0.4
        w = 0.25 * (decision % 3 - 1)
        return [v, w]

    def inp_stack(_, last, cur):
        # fa,fd,fc+1,fp+1 ORAZ fo doklejone na końcu
        inp = np.stack([cur[2], cur[3], cur[4], cur[5], last[2], last[3], last[4], last[5], cur[6], last[6]], axis=-1)
        return inp

    def decision(self, last, cur):
        inp = np.expand_dims(self.inp_stack(last, cur), axis=-1)
        inp = np.expand_dims(inp, axis=0)
        return self.model(inp).numpy().flatten()

    def app(self):
        env = TurtlesimEnvMulti()
        env.setup("routes_3.csv", agent_cnt=self.agent_cnt)
        agents = env.reset()
        env.MAX_STEPS = -1
        tnames = list(agents.keys())
        done_agents = set()
        from copy import deepcopy

        # Assign a different pen color to each agent
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 128),  # Purple
            (255, 128, 0),  # Orange
            (0, 128, 128),  # Teal
            (128, 128, 0),  # Olive
        ]
        for idx, tname in enumerate(tnames):
            r, g, b = colors[idx % len(colors)]
            set_pen_req = SetPenRequest(r=r, g=g, b=b, width=5, off=0)
            env.tapi.setPen(tname, set_pen_req)

        # Store last and current state for each agent
        last_states = {tname: deepcopy(agents[tname].map) for tname in tnames}
        current_states = {tname: deepcopy(agents[tname].map) for tname in tnames}
        done_cnt = 0

        while done_cnt < self.agent_cnt:
            controls = {}
            for tname in env.agents:
                controls[tname] = np.argmax(self.decision(last_states[tname], current_states[tname]))
            actions = {tname: self.ctl2act(control) for tname, control in controls.items()}
            scene = env.step(actions)
            for tname, (new_state, reward, done) in scene.items():
                last_states[tname] = current_states[tname]
                current_states[tname] = new_state
                if done:
                    done_cnt += 1



if __name__ == "__main__":
    app = AppMulti(
        "models/X6-c20c20c20d64-M-lr001-Gr5_Cr150_Sw0.5_Sv-10.0_Sf-4.0_Dr2.0_Oo-10_Cd1.5_Ms80_Pb3_D0.9_E0.99_e0.05_M20000_m2000_B32_U20_P500_T2.keras"
    )
    app.app()

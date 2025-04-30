import sys
import numpy as np
from tensorflow import keras
from copy import deepcopy
import turtlesim
from turtlesim.srv  import SetPenRequest
from turtlesim_env_single import TurtlesimEnvSingle


class AppSingle:
    def __init__(self, model: str):
        self.model = keras.models.load_model(model)
        self.set_pen = False

# zakodowanie wybranego sterowania (0-5) na potrzeby środowiska: (prędkość,skręt)
    def ctl2act(_,decision:int):            # prędkość\skręt    -.1rad 0 .1rad
        v = .2                              #   0.2                0   1   2
        if decision >= 3:                   #   0.4                3   4   5
            v = .4
        w=.25*(decision%3-1)
        return [v,w]

    # złożenie dwóch rastrów sytuacji aktualnej i poprzedniej w tensor 5x5x8 wejścia do sieci
    def inp_stack(_,last,cur):
        # fa,fd,fc+1,fp+1 - z wyjścia get_map - BEZ 2 POCZ. WARTOŚCI (zalecana prędkość w ukł. odniesienia planszy)
        inp = np.stack([cur[2],cur[3],cur[4],cur[5],last[2],last[3],last[4],last[5]], axis=-1)
        return inp

    # predykcja nagród łącznych (Q) za sterowania na podst. bieżącej i ostatniej sytuacji
    def decision(self,the_model,last,cur):
        inp=np.expand_dims(self.inp_stack(last,cur),axis=-1)
        inp=np.expand_dims(inp,axis=0)
        # return the_model.predict(inp,verbose=0).flatten() # wektor przewidywanych nagród dla sterowań -> UBYTEK PAMIĘCI w dockerze
        return the_model(inp).numpy().flatten()             # wektor przewidywanych nagród dla sterowań

    def app(self):
        env = TurtlesimEnvSingle()
        env.setup("routes.csv", agent_cnt=1)
        agents = env.reset()
        tname = list(agents.keys())[0]
        if self.set_pen:
            set_pen_req = turtlesim.srv.SetPenRequest(r=200, g=0, b=0, width=3, off=0)
            env.tapi.setPen(tname,set_pen_req)
        current_state = deepcopy(agents[tname].map)
        while not env.out_of_track:
            last_state = deepcopy(current_state)
            control = np.argmax(self.decision(self.model, last_state, current_state))
            current_state, _, _ = env.step({tname: self.ctl2act(control)},realtime=False)


if __name__ == "__main__":
    
    app = AppSingle("models/dqns-Gr5_Cr150_Sw0.5_Sv-15.0_Sf-4.0_Dr2.0_Oo-10_Cd1.5_Ms80_Pb6_D0.9_E0.99_e0.05_M20000_m400_B32_U5_P4000_T2episode20_model.keras")
    app.app()


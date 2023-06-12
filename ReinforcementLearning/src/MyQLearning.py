from QLearning import QLearning
import numpy as np
from State import State
from Action import Action

class MyQLearning(QLearning):
    def update_q(self, state:State, action:Action, r, state_next:State, possible_actions:np.ndarray, alfa, gamma):
        # TODO Auto-generated method stub
        cur_value = self.get_q(state, action)
        values = self.get_action_values(state_next, possible_actions)
        new_value = cur_value + alfa * (r + gamma * np.max(values) - cur_value)

        # set the value
        self.set_q(state, action, new_value)

        return

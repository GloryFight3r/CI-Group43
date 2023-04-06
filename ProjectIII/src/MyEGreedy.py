from Agent import Agent
from Action import Action
from Maze import Maze
from QLearning import QLearning
import numpy as np

class MyEGreedy:
    def __init__(self):
        print("Made EGreedy")

    def get_random_action(self, agent:Agent, maze:Maze):
        # TODO to select an action at random in State s
        actions = maze.get_valid_actions(agent)
        
        rnd = np.random.randint(len(actions))

        return actions[rnd]

    def get_best_action(self, agent:Agent, maze:Maze, q_learning:QLearning):
        # TODO to select the best possible action currently known in State s.
        actions = maze.get_valid_actions(agent)

        all_rewards = []
            
        for x in actions:
            newAgent = agent
            newAgent.do_action(x, maze)
            cur_reward = q_learning.get_q(newAgent.get_state(maze), x)
            all_rewards.append(cur_reward)

        return actions[np.argmax(all_rewards)]

    def get_egreedy_action(self, agent, maze, q_learning, epsilon):
        # TODO to select between random or best action selection based on epsilon.
        rnd = np.random.rand()
        if rnd < epsilon:
            return self.get_random_action(agent, maze)
        else:
            return self.get_best_action(agent, maze, q_learning)

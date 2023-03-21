import numpy as np

class AntColonyHistory:
    def __init__(self, generations: int):
        self.generations = generations
        self.shortest = np.empty(generations, dtype='int')
        self.success = np.empty(generations, dtype='float')
    def append_record(self, gen:int, shortest_length: int, success_percentage: float):
        self.shortest[gen] = shortest_length
        self.success[gen] = success_percentage

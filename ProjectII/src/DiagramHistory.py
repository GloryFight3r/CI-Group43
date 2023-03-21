import numpy as np

class AntColonyHistory:
    def __init__(self, generations: int):
        self.generations = generations
        self.shortest = np.ones(generations, dtype='int') * -1
        self.success = np.ones(generations, dtype='float') * -1

    def append_record(self, gen:int, shortest_length: int, success_percentage: float):
        self.shortest[gen] = shortest_length
        self.success[gen] = success_percentage
    
    def get_shortest_by_gen(self, gen):
        return self.shortest[gen]

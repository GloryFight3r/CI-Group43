import time
from Maze import Coordinate, Maze
from PathSpecification import PathSpecification
from Ant import Ant
import numpy as np 

# Class representing the first assignment. Finds shortest path between two points in a maze according to a specific
# path specification.
class AntColonyOptimization:

    # Constructs a new optimization object using ants.
    # @param maze the maze .
    # @param antsPerGen the amount of ants per generation.
    # @param generations the amount of generations.
    # @param Q normalization factor for the amount of dropped pheromone
    # @param evaporation the evaporation factor.
    def __init__(self, maze, ants_per_gen, generations, q, evaporation):
        self.maze = maze
        self.ants_per_gen = ants_per_gen
        self.generations = generations
        self.q = q
        self.evaporation = evaporation

     # Loop that starts the shortest path process
     # @param spec Spefication of the route we wish to optimize
     # @return ACO optimized route
    def find_shortest_route(self, path_specification):
        self.maze.reset()
        prevAvgLen = self.maze.length*self.maze.width
        route = None 
        ant = [None] * self.ants_per_gen
        routes = [None] * self.ants_per_gen
        #print(self.maze, self.maze.width, self.maze.length)
        #print(self.maze.walls)
        to_start = []
        for i in range(self.maze.width):
            for j in range(self.maze.length):
                if self.maze.walls[i][j] == 1:
                    to_start.append(Coordinate(j, i))
        real_start = path_specification.start
        for gen in range(self.generations):
            shortestLen = self.maze.length*self.maze.width
            curAvgLen = 0
            successful = 0
            pheromones = np.zeros(self.maze.width*self.maze.length*4)

            for x in range(self.ants_per_gen):
                path_specification.start = to_start[np.random.randint(len(to_start))]
                if gen > (self.generations * 9/10):
                    path_specification.start = real_start
                ant[x] = Ant(self.maze,path_specification)
                rt = ant[x].find_route()
                routes[x] = rt[0]
                locations = rt[1]
                sz = routes[x].size()
                if sz!=0:
                    #print("DAS")
                    pheromones[locations] += self.q/(len(locations))  # update the pheromones for this route
                    if path_specification.start == real_start:
                        shortestLen = min(sz,shortestLen)
                        if(sz == shortestLen):
                            route = routes[x]

                    #curAvgLen = curAvgLen + sz
                    #successful = successful + 1
            self.maze.evaporate(self.evaporation)
            self.maze.add_pheromone_routes(pheromones) 

        return route

import time
from Maze import Coordinate, Maze
from PathSpecification import PathSpecification
from Ant import Ant
import numpy as np 
from DiagramHistory import AntColonyHistory

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
    def find_shortest_route(self, path_specification, alpha: float, beta: float, random_start: float, toxic_start: float, convergence: int):
        self.maze.reset()
        route = None 
        ant = None
        routes = None

        to_start = []
        for i in range(self.maze.width):
            for j in range(self.maze.length):
                if self.maze.walls[i][j] == 1:
                    to_start.append(Coordinate(j, i))

        real_start = path_specification.start
        prevShortestLen = self.maze.length * self.maze.width
    
        colonyHistory = AntColonyHistory(self.generations)

        for gen in range(self.generations):
            shortestLen = self.maze.length*self.maze.width
            pheromones = np.zeros(self.maze.width*(self.maze.length*4))
            dead_trail = np.ones(self.maze.width*self.maze.length)
            successful_ends = 0

            for x in range(self.ants_per_gen):
                # ants start at random
                if gen < (self.generations * random_start):
                    path_specification.start = to_start[np.random.randint(len(to_start))]
                else:
                    path_specification.start = real_start

                ant = Ant(self.maze,path_specification)
                rt = ant.find_route(alpha, beta, dead_trail)
                routes = rt[0]
                locations = rt[1]
                is_dead = rt[2]
                sz = routes.size()
                if not is_dead:
                    pheromones[locations] += self.q/(len(locations))  # update the pheromones for this route
                    if path_specification.start == real_start:
                        shortestLen = min(sz,shortestLen)
                        successful_ends += 1
                        if(sz == shortestLen):
                            route = routes
                elif gen < int(self.generations * toxic_start):
                    dead_trail[locations] = np.minimum(dead_trail[locations], np.linspace(1, 0, len(locations)))

            if gen % convergence == 0:
                if shortestLen == prevShortestLen:
                    break
                prevShortestLen = shortestLen
            self.maze.evaporate(self.evaporation)
            self.maze.add_pheromone_routes(pheromones) 

            colonyHistory.append_record(gen, shortestLen, successful_ends / self.ants_per_gen)

        return (route, colonyHistory)

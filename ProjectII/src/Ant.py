import random
from Route import Route
import numpy as np
from Direction import Direction

#Class that represents the ants functionality.
class Ant:

    # Constructor for ant taking a Maze and PathSpecification.
    # @param maze Maze the ant will be running in.
    # @param spec The path specification consisting of a start coordinate and an end coordinate.
    def __init__(self, maze, path_specification):
        self.maze = maze
        self.start = path_specification.get_start()
        self.end = path_specification.get_end()
        self.current_position = self.start
        self.rand = random

    # Method that performs a single run through the maze by the ant.
    # @return The route the ant found through the maze.
    def find_route(self):
        route = Route(self.start)
        visited = np.zeros((self.maze.width,self.maze.length))
        dir = [Direction.east,Direction.north,Direction.west,Direction.south] 
        while self.current_position != self.end:
            visited[self.current_position.x][self.current_position.y] = 1 
            pheromones = self.maze.get_surrounding_pheromone(self.current_position)
            for j in range(4):
                possible = self.current_position.add_direction(dir[j])
                if self.maze.walls[possible.x][possible.y] == 0 or self.maze.in_bounds(possible) == False or visited[possible.x][possible.y] == 1:
                    pheromones[j] = 0 
            sm = np.sum(pheromones,axis=None)            
            if sm == 0:
                route = Route(self.start) 
                return [route,np.zeros(0, dtype='int')] 

            choice = random.choices(dir, weights = pheromones, k = 1)[0]
            self.current_position = self.current_position.add_direction(choice)
            route.add(choice)
        
        locations = np.zeros(route.size(), dtype='int')
        #print(locations)
        #print("DASDSA")
        cur_pos = self.start 
        for i,pos in enumerate(route.get_route()) : 
            locations[i] = cur_pos.x*self.maze.width + cur_pos.y + Direction.dir_to_int(pos)
            cur_pos = cur_pos.add_direction(pos)
        return [route,locations]

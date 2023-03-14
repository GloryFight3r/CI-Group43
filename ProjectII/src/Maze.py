import traceback
import sys
from Coordinate import Coordinate
from Direction import Direction
import numpy as np

# Class that holds all the maze data. This means the pheromones, the open and blocked tiles in the system as
# well as the starting and end coordinates.
class Maze:

    # Constructor of a maze
    # @param walls int array of tiles accessible (1) and non-accessible (0)
    # @param width width of Maze (horizontal)
    # @param length length of Maze (vertical)
    def __init__(self, walls, width, length):
        self.walls = walls
        self.length = length
        self.width = width
        self.pheromones_matrix = None
        self.start = None
        self.end = None
        self.initialize_pheromones()

    # Initialize pheromones to a start value.
    def initialize_pheromones(self):
        self.pheromones_matrix = np.zeros((self.length * self.width, self.length * self.width))
        for i in range(self.width):
            for j in range(self.length):
                position = Coordinate(i, j)
                N = position.add_direction(Direction.north)
                if self.in_bounds(N):
                    self.pheromones_matrix[self.regular_to_adj(N)][self.regular_to_adj(position)] = 1

                S = position.add_direction(Direction.south)
                if self.in_bounds(S):
                    self.pheromones_matrix[self.regular_to_adj(S)][self.regular_to_adj(position)] = 1

                E = position.add_direction(Direction.east)
                if self.in_bounds(E):
                    self.pheromones_matrix[self.regular_to_adj(E)][self.regular_to_adj(position)] = 1

                W = position.add_direction(Direction.west)
                if self.in_bounds(W):
                    self.pheromones_matrix[self.regular_to_adj(W)][self.regular_to_adj(position)] = 1

        # debug print
        print(self.pheromones_matrix.view())

    def regular_to_adj(self, position):
        return position.get_x() * self.length + position.get_y()

    # Reset the maze for a new shortest path problem.
    def reset(self):
        self.initialize_pheromones()

    # Update the pheromones along a certain route according to a certain Q
    # @param r The route of the ants
    # @param Q Normalization factor for amount of dropped pheromone
    def add_pheromone_route(self, route, q):
        walk_history = route.get_route()
        start = route.get_start()
        coords = []
        print(coords)
        coords.append(start)
        for i in range(len(walk_history)):
            coords.append(coords[-1].add_direction(walk_history[i]))
        print(coords)

        drop_value = q / len(walk_history)
        for i in range(1, len(coords)):
            self.pheromones_matrix[self.regular_to_adj(coords[i])][self.regular_to_adj(coords[i - 1])] += drop_value
        return

     # Update pheromones for a list of routes
     # @param routes A list of routes
     # @param Q Normalization factor for amount of dropped pheromone
    def add_pheromone_routes(self, routes, q):
        for r in routes:
            self.add_pheromone_route(r, q)

    # Evaporate pheromone
    # @param rho evaporation factor
    def evaporate(self, rho):
       self.pheromones_matrix *= (1 - rho)

    # Width getter
    # @return width of the maze
    def get_width(self):
        return self.width

    # Length getter
    # @return length of the maze
    def get_length(self):
        return self.length

    # Returns a the amount of pheromones on the neighbouring positions (N/S/E/W).
    # @param position The position to check the neighbours of.
    # @return the pheromones of the neighbouring positions.
    def get_surrounding_pheromone(self, position):
        N = position.add_direction(Direction.north)
        S = position.add_direction(Direction.south)
        E = position.add_direction(Direction.east)
        W = position.add_direction(Direction.west)

        return {
            Direction.north : get_pheromone(N),
            Direction.south : get_pheromone(S),
            Direction.east : get_pheromone(E),
            Direction.west : get_pheromone(W),
        }

    # Pheromone getter for a specific position. If the position is not in bounds returns 0
    # @param pos Position coordinate
    # @return pheromone at point
    def get_pheromone(self, pos):
        if not self.in_bounds(pos):
            return 0
        return self.get_surrounding_pheromone(pos)

    # Check whether a coordinate lies in the current maze.
    # @param position The position to be checked
    # @return Whether the position is in the current maze
    def in_bounds(self, position):
        return position.x_between(0, self.width) and position.y_between(0, self.length)

    # Representation of Maze as defined by the input file format.
    # @return String representation
    def __str__(self):
        string = ""
        string += str(self.width)
        string += " "
        string += str(self.length)
        string += " \n"
        for y in range(self.length):
            for x in range(self.width):
                string += str(self.walls[x][y])
                string += " "
            string += "\n"
        return string

    # Method that builds a mze from a file
    # @param filePath Path to the file
    # @return A maze object with pheromones initialized to 0's inaccessible and 1's accessible.
    @staticmethod
    def create_maze(file_path):
        try:
            f = open(file_path, "r")
            lines = f.read().splitlines()
            dimensions = lines[0].split(" ")
            width = int(dimensions[0])
            length = int(dimensions[1])
            
            #make the maze_layout
            maze_layout = []
            for x in range(width):
                maze_layout.append([])
            
            for y in range(length):
                line = lines[y+1].split(" ")
                for x in range(width):
                    if line[x] != "":
                        state = int(line[x])
                        maze_layout[x].append(state)
            print("Ready reading maze file " + file_path)
            return Maze(maze_layout, width, length)
        except FileNotFoundError:
            print("Error reading maze file " + file_path)
            traceback.print_exc()
            sys.exit()
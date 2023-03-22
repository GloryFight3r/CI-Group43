import pickle
import re
import sys
import traceback
from multiprocessing import Process
from multiprocessing import Queue
from Coordinate import Coordinate
from PathSpecification import PathSpecification

# Class containing the product distances. Can be either build from a maze, a product
# location list and a PathSpecification or be reloaded from a file.
class TSPData:

    # Constructs a new TSP data object.
    # @param productLocations the productlocations.
    # @param spec the path specification.
    def __init__(self, product_locations, spec):
        self.product_locations = product_locations
        self.spec = spec

        self.distances = None
        self.start_distances = None
        self.end_distances = None
        self.product_to_product = None
        self.start_to_product = None
        self.product_to_end = None

    # Calculate the routes from the product locations to each other, the start, and the end.
    # Additionally generate arrays that contain the length of all the routes.
    # @param maze
    def calculate_routes(self, aco, alpha, random_start, toxic_start, convergence, alpha_ants):
        self.product_to_product = self.build_distance_matrix(aco, alpha, random_start, toxic_start, convergence, alpha_ants)
        self.start_to_product = self.build_start_to_products(aco, alpha, random_start, toxic_start, convergence, alpha_ants)
        self.product_to_end = self.build_products_to_end(aco, alpha, random_start, toxic_start, convergence, alpha_ants)
        self.build_distance_lists()
        return

    # Build a list of integer distances of all the product-product routes.
    def build_distance_lists(self):
        number_of_products = len(self.product_locations)
        self.distances = []
        self.start_distances = []
        self.end_distances = []

        for i in range(number_of_products):
            self.distances.append([])
            for j in range(number_of_products):
                self.distances[i].append(self.product_to_product[i][j].size())
            self.start_distances.append(self.start_to_product[i].size())
            self.end_distances.append(self.product_to_end[i].size())
        return

    # Distance product to product getter
    # @return the list
    def get_distances(self):
        return self.distances

    # Distance start to product getter
    # @return the list
    def get_start_distances(self):
        return self.start_distances

    # Distance product to end getter
    # @return the list
    def get_end_distances(self):
        return self.end_distances

    # Equals method
    # @param other other TSPData to check
    # @return boolean whether equal
    def __eq__(self, other):
        return self.distances == other.distances \
               and self.product_to_product == other.product_to_product \
               and self.product_to_end == other.product_to_end \
               and self.start_to_product == other.start_to_product \
               and self.spec == other.spec \
               and self.product_locations == other.product_locations

    # Persist object to file so that it can be reused later
    # @param filePath Path to persist to
    def write_to_file(self, file_path):
        pickle.dump(self, open(file_path, "wb"))

    # Write away an action file based on a solution from the TSP problem.
    # @param productOrder Solution of the TSP problem
    # @param filePath Path to the solution file
    def write_action_file(self, product_order, file_path):
        total_length = self.start_distances[product_order[0]]
        for i in range(len(product_order) - 1):
            frm = product_order[i]
            to = product_order[i + 1]
            total_length += self.distances[frm][to]

        total_length += self.end_distances[product_order[len(product_order) - 1]] + len(product_order)

        string = ""
        string += str(total_length)
        string += ";\n"
        string += str(self.spec.get_start())
        string += ";\n"
        string += str(self.start_to_product[product_order[0]])
        string += "take product #"
        string += str(product_order[0] + 1)
        string += ";\n"

        for i in range(len(product_order) - 1):
            frm = product_order[i]
            to = product_order[i + 1]
            string += str(self.product_to_product[frm][to])
            string += "take product #"
            string += str(to + 1)
            string += ";\n"
        string += str(self.product_to_end[product_order[len(product_order) - 1]])

        f = open(file_path, "w")
        f.write(string)

    # Calculate the optimal routes between all the individual routes
    # @param maze Maze to calculate optimal routes in
    # @return Optimal routes between all products in 2d array
    def build_distance_matrix(self, aco, alpha, random_start, toxic_start, convergence, alpha_ants):
        number_of_product = len(self.product_locations)
        product_to_product = []
        
        my_thread = []
        queue = Queue()
        for i in range(number_of_product):
            product_to_product.append([])
            for j in range(number_of_product):
                start = self.product_locations[i]
                end = self.product_locations[j]
                my_thread.append(Process(target=aco.find_shortest_route, args=[PathSpecification(start, end), alpha, random_start, toxic_start, convergence, alpha_ants, queue]))

        for cur_thread in my_thread:
            cur_thread.start()

        for ind, cur_thread in enumerate(my_thread):
            product_to_product[int(ind/number_of_product)].append(queue.get())#cur_thread.join()[0])

        return product_to_product


    # Calculate optimal route between the start and all the products
    # @param maze Maze to calculate optimal routes in
    # @return Optimal route from start to products
    def build_start_to_products(self, aco, alpha, random_start, toxic_start, convergence, alpha_ants):
        start = self.spec.get_start()
        start_to_products = []
        my_thread = []
        queue = Queue()
        for i in range(len(self.product_locations)):
            my_thread.append(Process(target=aco.find_shortest_route, args=[PathSpecification(start, self.product_locations[i]), alpha, random_start, toxic_start, convergence, alpha_ants, queue]))
            #my_thread.append(Process(target=aco.find_shortest_route, args=([[PathSpecification(start, self.product_locations[i])]], alpha, random_start, toxic_start, convergence, alpha_ants,)))

        for cur_thread in my_thread:
            cur_thread.start()

        for ind, cur_thread in enumerate(my_thread):
            start_to_products.append(queue.get())#cur_thread.join()[0])

        return start_to_products

    # Calculate optimal routes between the products and the end point
    # @param maze Maze to calculate optimal routes in
    # @return Optimal route from products to end
    def build_products_to_end(self, aco, alpha, random_start, toxic_start, convergence, alpha_ants):
        end = self.spec.get_end()
        products_to_end = []
        my_thread = []
        queue = Queue()
        for i in range(len(self.product_locations)):    
            my_thread.append(Process(target=aco.find_shortest_route, args=[PathSpecification(self.product_locations[i], end), alpha, random_start, toxic_start, convergence, alpha_ants, queue]))
            #my_thread.append(Process(target=aco.find_shortest_route, args=([[PathSpecification(self.product_locations[i], end)]], alpha, random_start, toxic_start, convergence, alpha_ants,)))
        
        for cur_thread in my_thread:
            cur_thread.start()

        for ind, cur_thread in enumerate(my_thread):
            products_to_end.append(queue.get())#cur_thread.join()[0])

        return products_to_end

    # Load TSP data from a file
    # @param filePath Persist file
    # @return TSPData object from the file
    @staticmethod
    def read_from_file(file_path):
        return pickle.load(open(file_path, "rb"))

    # Read a TSP problem specification based on a coordinate file and a product file
    # @param coordinates Path to the coordinate file
    # @param productFile Path to the product file
    # @return TSP object with uninitiatilized routes
    @staticmethod
    def read_specification(coordinates, product_file):
        try:
            f = open(product_file, "r")
            lines = f.read().splitlines()

            firstline = re.compile("[:,;]\\s*").split(lines[0])
            product_locations = []
            number_of_products = int(firstline[0])
            for i in range(number_of_products):
                line = re.compile("[:,;]\\s*").split(lines[i + 1])
                product = int(line[0])
                x = int(line[1])
                y = int(line[2])
                product_locations.append(Coordinate(x, y))
            spec = PathSpecification.read_coordinates(coordinates)
            return TSPData(product_locations, spec)
        except FileNotFoundError:
            print("Error reading file " + product_file)
            traceback.print_exc()
            sys.exit()

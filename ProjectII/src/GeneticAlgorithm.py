import random
import numpy as np
from TSPData import TSPData

class Candidate:
    def __init__(self, chromosome: np.ndarray, maxBit: int):
        self.maxBit = maxBit
        self.chromosome = self.encode(chromosome)
        
    def encode(self, chromosome: np.ndarray):
        encoded = []
        for i in chromosome:
            encoded.append(bin(i)[-self.maxBit:])
        return encoded

    def decode(self):
        return_array = np.zeros(len(self.chromosome))
        for ind, cur_number in enumerate(self.chromosome):
            cur_number = int(current, '2')
            return_array[ind] = cur_number
        return return_array
        
    def calculate_fitness(self, tsp_data: TSPData):
        prev_number = -1
        cur_fitness = 0
        for cur_number in self.decode():
            if prev_number != -1:
                cur_fitness += tsp_data.get_distances()[prev_number][cur_number]
            prev_number = cur_number
        return cur_fitness

    def produce_offspring(self, other_chromosome: Candidate):
        pass

    def cross_over(self, other_chromosome: Candidate):
        pass
    def mutate(self, p: float):
        pass

# TSP problem solver using genetic algorithms.
class GeneticAlgorithm:
    
    # Constructs a new 'genetic algorithm' object.
    # @param generations the amount of generations.
    # @param popSize the population size.
    def __init__(self, generations, pop_size):
        self.generations = generations
        self.pop_size = pop_size

    def encode_data(self, k: int, pop_size: int, maxBit : int) -> list[Candidate]:
        # they start at 0
        encodings = []
        for current_candidate in range(pop_size):
            current_perm = np.random.permutation(k)
            candidate = Candidate(current_perm, maxBit)

            encodings.append(candidate)
        return encodings

    def calculate_fitness(self, population: list[Candidate], tsp_data: TSPData) -> np.ndarray:
        fitness_array = np.zeros(len(population))
        for current_candidate in population:
            current_candidate.calculate_fitness(tsp_data)
        return fitness_array

    def produce_offspring(self, population: list[Candidate], fitness_ratio: np.ndarray) -> list[Candidate]:
        fitness_raitio = enumerate(fitness_ratio)

        for pop_to_produce in self.pop_size:
            while True:
                first_candidate = 0
                second_candidate = 0


        pass

    # This method should solve the TSP.
    # @param pd the TSP data.
    # @return the optimized product sequence.
    def solve_tsp(self, tsp_data: TSPData):
        n_items = len(tsp_data.get_distances())
        # choose initial population
        population = self.encode_data(n_items, self.pop_size, np.log2(n_items) + 1)

        for i in self.pop_size:
            # make offspring
            # calculate fitness function for the current population
            fitness = self.calculate_fitness(population, tsp_data)
            fitness_ratio = fitness / np.sum(fitness)
            population = produce_offspring(population, fitness_ratio)

        return []

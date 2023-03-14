import random
import numpy as np
import copy
from TSPData import TSPData

class Candidate:
    def __init__(self, chromosome: np.ndarray, maxBit: int):
        self.maxBit = maxBit
        self.chromosome = self.encode(chromosome)
        
    def encode(self, chromosome: np.ndarray):
        encoded = []
        for i in chromosome:
            to_str = bin(i)[2:]
            to_str = ('0' * (self.maxBit - len(to_str))) + to_str

            encoded.append(to_str)
        return encoded

    def decode(self)->np.ndarray:
        return_array = np.zeros(len(self.chromosome), dtype='int')
        for ind, number in enumerate(self.chromosome):
            cur_number = int(number, 2)
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

    def is_legal(self)->bool:
        data = self.decode()
        k = len(data)
        data = np.array(list(filter(lambda x : x < k, data)))
        seen = np.zeros(k, dtype=bool)
        seen[data] = True
        return np.all(seen)

    def produce_offspring(self, other_chromosome, cross_over_prob: float, mutation_prob: float):
        cross_over_chance = np.random.rand()

        new_cand = copy.deepcopy(self)

        if(cross_over_chance <= cross_over_prob):
            new_cand.cross_over(other_chromosome)

        new_cand.mutate(mutation_prob)

        return new_cand

    def cross_over(self, other_chromosome):
        start_from = np.random.randint(len(self.chromosome))

        self.chromosome[start_from:] = other_chromosome.chromosome[start_from:]
    def mutate(self, p: float):
        for ind, gene in enumerate(self.chromosome):
            indeces = np.where(np.random.rand(self.maxBit) <= p)[0]
            newGene = gene
            for z in indeces:
                newGene = newGene[:z] + ('1' if newGene[z] == '0' else '0') + newGene[z + 1:]
            self.chromosome[ind] = newGene
        #print(self.chromosome)
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
        for ind, current_candidate in enumerate(population):
            fitness_array[ind] = current_candidate.calculate_fitness(tsp_data)
        return fitness_array

    def pick_candidate(self, fitness_ratio, prob: float):
        #print(fitness_ratio)
        #print(np.where(fitness_ratio > prob)[0][0], prob)
        return np.where(fitness_ratio > prob)[0][0]

    def produce_offspring(self, population: list[Candidate], fitness_ratio: np.ndarray, c_p: float, m_p: float) -> list[Candidate]:
        fitness_ratio = list(np.ndenumerate(fitness_ratio))
        fitness_ratio_with_ind = sorted(fitness_ratio, key = lambda x : x[1])
        fitness_ratio = np.cumsum(np.array(list(map(lambda z : z[1], fitness_ratio_with_ind))))

        #print(fitness_ratio)

        offspring = []

        for pop_to_produce in range(self.pop_size):
            while True:
                first_ind = self.pick_candidate(fitness_ratio, np.random.rand())
                second_ind = self.pick_candidate(fitness_ratio, np.random.rand())
                real_first_ind = fitness_ratio_with_ind[first_ind][0][0]
                real_second_ind = fitness_ratio_with_ind[second_ind][0][0]
                
                first_candidate:Candidate = population[real_first_ind]
                second_candidate:Candidate = population[real_second_ind]
                
                new_offspring = first_candidate.produce_offspring(second_candidate, c_p, m_p)
                if new_offspring.is_legal():
                    offspring.append(new_offspring)
                    break

        return offspring

    # This method should solve the TSP.
    # @param pd the TSP data.
    # @return the optimized product sequence.
    def solve_tsp(self, tsp_data: TSPData, cross_over_prob, mutation_prob):
        n_items = len(tsp_data.get_distances())
        # choose initial population
        #print(n_items, int(np.log2(n_items)) + 1)
        population = self.encode_data(n_items, self.pop_size, int(np.log2(n_items)) + 1)
        bst = 1000
        for i in range(self.pop_size):
            # make offspring
            # calculate fitness function for the current population
            fitness = self.calculate_fitness(population, tsp_data)
            #print(fitness)
            max_value = np.max(fitness) + 0.001
            min_value = np.min(fitness)
    
            bst = min_value
            
            fitness_ratio = ((max_value - fitness) / (max_value - min_value))
            fitness_ratio = fitness_ratio / np.sum(fitness_ratio)
            #print(fitness_ratio)
            population = self.produce_offspring(population, fitness_ratio, cross_over_prob, mutation_prob)
        print(bst)
        return []

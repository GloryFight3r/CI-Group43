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
        product_order = self.decode()
        total_length = tsp_data.start_distances[product_order[0]]
        for i in range(len(product_order) - 1):
            frm = product_order[i]
            to = product_order[i + 1]
            total_length += tsp_data.distances[frm][to]

        total_length += tsp_data.end_distances[product_order[len(product_order) - 1]] + len(product_order)
        return total_length

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
            oth_chrom = copy.deepcopy(other_chromosome)
            new_cand.cross_over(oth_chrom)

            oth_chrom.mutate(mutation_prob)
            new_cand.mutate(mutation_prob)
            return [new_cand, oth_chrom]

        new_cand.mutate(mutation_prob)

        return [new_cand]

    def cross_over(self, other_chromosome):
        start_from = np.random.randint(len(self.chromosome))

        self.chromosome[start_from:] = other_chromosome.chromosome[start_from:]
        other_chromosome.chromosome[:start_from] = self.chromosome[:start_from]
    def mutate(self, p: float):
        # swapping genes mutation 
        """
        For every gene we mark it with probability p and then random shuffle all the marked ones
        """
        indeces = np.where(np.random.rand(len(self.chromosome)) <= p)[0]
        cpy = copy.deepcopy(self.chromosome)
        other_indeces = copy.deepcopy(indeces)

        np.random.shuffle(other_indeces)
        for ind, i in enumerate(indeces):
            self.chromosome[i] = cpy[other_indeces[ind]]
        
        # flipping bit mutation
        #for ind, gene in enumerate(self.chromosome):
        #    indeces = np.where(np.random.rand(self.maxBit) <= p)[0]
        #    newGene = gene
        #    for z in indeces:
        #        newGene = newGene[:z] + ('1' if newGene[z] == '0' else '0') + newGene[z + 1:]
        #    self.chromosome[ind] = newGene

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
        return np.where(fitness_ratio > prob)[0][0]

    def produce_offspring(self, population: list[Candidate], fitness_ratio: np.ndarray, c_p: float, m_p: float) -> list[Candidate]:
        fitness_ratio = list(np.ndenumerate(fitness_ratio))
        fitness_ratio_with_ind = sorted(fitness_ratio, key = lambda x : x[1])
        fitness_ratio = np.cumsum(np.array(list(map(lambda z : z[1], fitness_ratio_with_ind))))

        offspring = []

        while(len(offspring) < self.pop_size):
            first_ind = self.pick_candidate(fitness_ratio, np.random.rand())
            second_ind = self.pick_candidate(fitness_ratio, np.random.rand())
            real_first_ind = fitness_ratio_with_ind[first_ind][0][0]
            real_second_ind = fitness_ratio_with_ind[second_ind][0][0]
                
            first_candidate:Candidate = population[real_first_ind]
            second_candidate:Candidate = population[real_second_ind]
                
            for cnd in first_candidate.produce_offspring(second_candidate, c_p, m_p):
                if cnd.is_legal():
                    offspring.append(cnd)

        return offspring[:self.pop_size]

    # This method should solve the TSP.
    # @param pd the TSP data.
    # @return the optimized product sequence.
    def solve_tsp(self, tsp_data: TSPData, cross_over_prob, mutation_prob):
        n_items = len(tsp_data.get_distances())
        # choose initial population
        population = self.encode_data(n_items, self.pop_size, int(np.log2(n_items)) + 1)
        fitness = []
        history = []
        for i in range(self.generations):
            # make offspring
            # calculate fitness function for the current population
            fitness = self.calculate_fitness(population, tsp_data)
            max_value = np.max(fitness) + 0.001
            min_value = np.min(fitness)
    
            bst = min_value
            history.append(History(np.mean(fitness), max_value, min_value))

            #fitness_ratio = 1 / fitness
            #fitness_ratio = fitness_ratio / np.sum(fitness_ratio)
            
            #print(fitness_ratio)
            fitness_ratio = ((max_value - fitness) / (max_value - min_value))
            fitness_ratio = fitness_ratio / np.sum(fitness_ratio)
            population = self.produce_offspring(population, fitness_ratio, cross_over_prob, mutation_prob)
        return (population[np.argmin(fitness)].decode(), history)

class History:
    def __init__(self, average, max, min):
        self.average = average
        self.max = max
        self.min = min

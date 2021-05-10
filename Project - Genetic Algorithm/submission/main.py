
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import json
import time
# import sys

import client as server
# %matplotlib inline

class GeneticAlgorithm:
    '''
    class for genetic algorithm model
    '''
    
    OVERFIT_GENE = [0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,  
                8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10]
    
    # the length of a gene (weight vector)
    GENE_LENGTH = 11

    # the number of genes present in a generation
    POPULATION_SIZE = 8
    
    # limits for each weight (chromosome)
    GENE_MIN = -10
    GENE_MAX = 10

    # probability of mutation of a single chromosome
    MUTATE_PROB = 0.6
    
    # scaling the value to be added for mutation, greater means less variation to mutated
    MUTATE_NORMALIZER = 10
    OVERFIT_MUTATE_NORMALIZER = 0.1

    # number of generations for which the model trains
    NUM_GENERATIONS = 25

    def __init__(self, gene: list, is_zero = 0):
        
        if len(gene) != self.GENE_LENGTH:
            raise ValueError
            
        if is_zero:
            self.population = self.generate_population_random(gene)
        else:
            self.population = self.generate_population(gene)

        self.log_dict = {}
#         self.print_population(self.population)
        
        self.log_dict["POPULATION_SIZE"] = self.POPULATION_SIZE
        self.log_dict["NUM_GENERATIONS"] = self.NUM_GENERATIONS
        self.log_dict["MUTATE_NORMALIZER"] = self.MUTATE_NORMALIZER
        self.log_dict["MUTATE_PROB"] = self.MUTATE_PROB
        self.max_overall_fitness = None
        self.best_overall_gene = None
        self.train_for_best_gene = None
        self.valid_for_best_gene = None
        self.avg_fitness = []                          # maintained across generations
        self.avg_train_errors = []                     # maintained across generations
        self.avg_validation_errors = []                # maintained across generations
        self.gene_and_fitness = []                   # consists of tuples (vector,fitness)
        
    def print_population(self, population):
        for gene in population:
            print(gene)
            
            
    @classmethod
    def mutate(self, population):
        '''
        Mutates the population randomly
        '''
        def add_uniform_noise(population):
            for idx, val in np.ndenumerate(population):
                if np.random.random() < self.MUTATE_PROB:
                    range_lim = abs(val/self.MUTATE_NORMALIZER)
                    noise = np.random.uniform(-range_lim, range_lim)
                    
                    if noise == 0:
                        noise = (random.random() - 0.5) / 1e13
                    
                    population[idx] = population[idx] + noise
            mutated = np.clip(population, self.GENE_MIN, self.GENE_MAX)
            return mutated
        
        def add_overfit_noise(population):
            for idx, val in np.ndenumerate(population):
                if np.random.random() < self.MUTATE_PROB:
                    range_lim = abs(self.OVERFIT_GENE[idx[1]] * self.OVERFIT_MUTATE_NORMALIZER)
                    noise = np.random.uniform(-range_lim, range_lim)
                    population[idx] = population[idx] + noise
            mutated = np.clip(population, self.GENE_MIN, self.GENE_MAX) 
            return mutated
                
#         mutated = add_uniform_noise(population)  
        mutated = add_overfit_noise(population)  
        return mutated

    def generate_population(self, gene: list):
    # generating a population from a single list of gene

        pop = [list(gene) for i in range(self.POPULATION_SIZE)]
        pop =  self.mutate(np.array(pop, dtype=np.double))
        pop[0] = gene
        pop = np.array(pop)
        return pop
    
    def generate_population_random(self, gene: list):
    # generating a random population from a single list of gene

        temp = []
        for i in range(self.POPULATION_SIZE):
            temp.append([(random.random()  - 0.5) / (1e12) for x in range(self.GENE_LENGTH)])
            
        temp = np.array(temp, dtype=np.double)
        temp[0] = gene
        return np.array(temp)
        
        
    @classmethod
    def crossover(self, mom: np.ndarray, dad: np.ndarray):
    # for performing the crossover between two parents
        
        
        def uniform_crossover(mom, dad):
            # perform uniform crossover
            child1 = copy.deepcopy(mom)
            child2 = copy.deepcopy(dad)
            parents = np.array([mom,dad])
            for i in range(len(mom)):
                choose = np.random.randint(0,2)
                child1[i] = parents[choose][i]
                child2[i] = parents[1-choose][i]
                children = child1, child2

            return children
    
        def double_point(mom: np.ndarray, dad: np.ndarray):
            # perform double point crossover
            thresh = np.random.randint(self.GENE_LENGTH//2)
            thresh2 = np.random.randint(self.GENE_LENGTH//2+1,self.GENE_LENGTH) 

            child1 = copy.deepcopy(dad)
            child2 = copy.deepcopy(mom)    
            child1[thresh:thresh2] = mom[thresh:thresh2]
            child2[thresh:thresh2] = dad[thresh:thresh2]
            children = child1, child2

            return children
        
        def single_point(mom: np.ndarray, dad: np.ndarray):
            # perform single point crossover
            thresh = np.random.randint(2,self.GENE_LENGTH-2)

            child1 = copy.deepcopy(dad)
            child1[0:thresh] = mom[0:thresh]
            
            child2 = copy.deepcopy(mom)
            child2[0:thresh] = dad[0:thresh]
            children = child1, child2

            return children

        return single_point(mom, dad)
#         return uniform_crossover(mom, dad)
#         return double_point(mom, dad)

    def get_fitness(self,num_gen):
    # get the fitness of population vectors

        def error_to_fitness(train_err, valid_err):
#             return -(5*valid_err + 2*abs(train_err - valid_err))
#             return -(train_err + valid_err)
#             return -(valid_err + 3*abs(valid_err - train_err))
#             return 1/( abs(valid_err-train_err) + 5*valid_err )
#             return 1/(train_err + 5*valid_err)
#             return 1 / (abs(train_err - valid_err))
            return 1 / (2*abs(train_err - valid_err) + 1.25*train_err + 3*valid_err)
#             return 1 / (2*abs(train_err - valid_err) + 5*valid_err + 2*train_err)

        fitness = []
        train_errors = []
        valid_errors = []
        weight_fitness = []
        counter = 1
        
        for gene in self.population:
            
            train_err, valid_err = server.get_errors(TEAM_ID, list(gene))
            fit = error_to_fitness(train_err, valid_err)                
            
            fitness.append(fit)
            train_errors.append(train_err)
            valid_errors.append(valid_err)
            weight_fitness.append((gene,fit))
        
            curr_dic = {}
            curr_dic["gene"] = gene.tolist()
            curr_dic["fitness"] = fit
            curr_dic["train_err"] = train_err
            curr_dic["valid_err"] = valid_err
                        
            self.log_dict["generation_"+str(num_gen)][counter] = copy.deepcopy(curr_dic)
            counter += 1
            
        fitness = np.array(fitness, dtype=np.double)
        self.gene_and_fitness = weight_fitness
        
        return fitness, train_errors, valid_errors
    

    def breed(self,num_gen):
        # for making the next generation
        def Sort_Tuple(tup):
            tup.sort(key = lambda x: x[1])  
            return tup  
        
        def normal_breed(num_gen):
            fitness, train_errors, valid_errors = self.get_fitness(num_gen)
            self.gene_and_fitness = Sort_Tuple(self.gene_and_fitness)
            self.gene_and_fitness.reverse()
            
            self.avg_fitness.append(np.mean(fitness))
            self.avg_train_errors.append(np.mean(train_errors))
            self.avg_validation_errors.append(np.mean(valid_errors))
            offsprings = []
            selected = []

            self.update_best(fitness, train_errors, valid_errors)
            
            CHOICE = 3

            for p1 in range(CHOICE):
                for p2 in range(p1+1,CHOICE):
                    mom = self.gene_and_fitness[p1][0]
                    selected.append(mom)
                    dad = self.gene_and_fitness[p2][0]                    
                    selected.append(dad)
                    child1,child2 = self.crossover(mom,dad)
                    offsprings.append(child1)
                    offsprings.append(child2)
            
            # keeping the best i parents as it is in the next gen
            for i in range(2):
                offsprings.append(self.gene_and_fitness[i][0])
                selected.append(self.gene_and_fitness[i][0])
                
            return np.array(selected, dtype=np.double),np.array(offsprings, dtype=np.double)
        
        sorted_population = []
        for vec in self.gene_and_fitness:
            sorted_population.append(vec[0].tolist())
        
        self.log_dict["generation_"+str(num_gen)]["sorted_population"] = sorted_population

#         offsprings = russian_roulette(num_gen)
        selected, offsprings = normal_breed(num_gen)

        self.log_dict["generation_"+str(num_gen)]["selected"] = selected.tolist()
        self.log_dict["generation_"+str(num_gen)]["offsprings"] = offsprings.tolist()

        self.population = self.mutate(offsprings)
        self.log_dict["generation_"+str(num_gen)]["mutated"] = self.population.tolist()

    def update_best(self, fitness: np.ndarray, train_errors: list, valid_errors: list):
        # Updates the best chromosome across generations parameter from self.population
        
        best_idx = np.argmax(fitness)
        if (self.max_overall_fitness) and  (fitness[best_idx] <= self.max_overall_fitness):
            return 

        self.max_overall_fitness = fitness[best_idx]
        self.log_dict["max_overall_fitness"] = self.max_overall_fitness

        self.best_overall_gene = self.population[best_idx]            
        self.log_dict["best_overall_gene"] = self.best_overall_gene.tolist()

        self.train_for_best_gene = train_errors[best_idx]
        self.log_dict["train_for_best_gene"] = self.train_for_best_gene

        self.valid_for_best_gene = valid_errors[best_idx]
        self.log_dict["valid_for_best_gene"] = self.valid_for_best_gene


    def log_data(self):
        
        # for filename
        localtime = str(time.asctime( time.localtime(time.time()) ))
        localtime = localtime.replace("  "," ")
        localtime = localtime.replace(":","-")
        localtime = localtime.replace(" ","_")
        
        # saving to a file
        file_name = localtime + ".json"
        out_file = open("logs/"+file_name, "w") 
        json.dump(self.log_dict, out_file, indent = 4) 
        out_file.close() 
        return localtime
        
    def train(self):
        self.log_dict["Time"] = str(time.asctime( time.localtime(time.time()) ))
        for i in range(self.NUM_GENERATIONS):
            # for vector in self.population:
            #     print((vector).tolist())
#             print("\n\nGeneration - ",i+1)
            self.log_dict["generation_"+str(i+1)] = {}
            self.breed(i+1)
        
        # saves to a file
        filename = self.log_data()
        
        fig, axs = plt.subplots(3, figsize=(10, 15))
        
        axs[0].plot(self.avg_fitness)
        axs[0].set_xlabel('Generations', fontsize=12)
        axs[0].set_ylabel('Best Fitness', fontsize=12)
        axs[0].set_title('Best Fitness across Generations', fontsize=14)

        axs[1].plot(self.avg_train_errors)
        axs[1].set_xlabel('Generations', fontsize=12)
        axs[1].set_ylabel('train errors', fontsize=12)
        axs[1].set_title('train error across generations', fontsize=14)
        
        axs[2].plot(self.avg_validation_errors)
        axs[2].set_xlabel('Generations', fontsize=12)
        axs[2].set_ylabel('validation errors', fontsize=12)
        axs[2].set_title('validation error across generations', fontsize=14)
    
        fig.savefig('logs/' + filename + '.png')
    
        return self.best_overall_gene, self.max_overall_fitness, self.train_for_best_gene, self.valid_for_best_gene







TEAM_ID = 'gCA6CLYnqUty40i93xmxPiCuNFk8wRA2wjui6iqDFxtDgHxhnb' # Room543

# the initial weights, that overfit the data
OVERFIT_GENE = [0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,  
                8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10]

model = GeneticAlgorithm(OVERFIT_GENE)

input("Press enter to continue for training the model")

best_gene, final_fitness, train_err, valid_err = model.train()

best_gene = best_gene.tolist()


# input("Press enter to continue for submitting the vector to leaderboard")

# print(server.submit(TEAM_ID, best_gene))


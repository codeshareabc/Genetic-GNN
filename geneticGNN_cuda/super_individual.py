import numpy as np

class Super_Individual(object):
    
    def __init__(self, args, population, param_genes):
        
        self.args = args
        self.population = population
        self.param_genes = param_genes

        
    def get_population(self):
        return self.population
    
    def get_param_genes(self):
        return self.param_genes
    
    def get_fitness(self):
        return self.fitness


    def cal_superfitness(self):
        best_fitness = -1
        average_fitness = 0
        for elem in self.population:
            if best_fitness < elem.get_fitness():
                best_fitness = elem.get_fitness()
            average_fitness += elem.get_fitness()
        
        average_fitness = average_fitness / self.args.num_individuals
        final_fitness = self.args.super_ratio * best_fitness + (1-self.args.super_ratio) * average_fitness
        
        self.fitness = final_fitness
    
    def set_newparam(self, param_genes):
        for elem in self.population:
            elem.param_genes = param_genes
    
    def mutation_net_gene(self, mutate_point, new_gene, type='struct'):
        if type == 'struct':
            self.net_genes[mutate_point] = new_gene
        elif type == 'param':
            self.param_genes[mutate_point] = new_gene
        else:
            raise Exception("wrong mutation type")
        
        

from search_space import HybridSearchSpace
from individual import Individual
from super_individual import Super_Individual
from random import sample, choices
import numpy as np
from gnn_model_manager import GNNModelManager
import copy

class Population(object):
    
    def __init__(self, args):
        
        self.args = args
        hybrid_search_space = HybridSearchSpace(self.args.num_gnn_layers)
        self.hybrid_search_space = hybrid_search_space
        
        # prepare data set for training the gnn model
        self.load_trining_data()
    
    def load_trining_data(self):
        self.gnn_manager = GNNModelManager(self.args)
        self.gnn_manager.load_data(self.args.dataset)
        
        # dataset statistics
        print(self.gnn_manager.data)
        
    def init_population(self):
        
        struct_individuals = []
        
        for i in range(self.args.num_individuals):
            net_genes = self.hybrid_search_space.get_net_instance()
#             param_genes = self.hybrid_search_space.get_param_instance()
            param_genes = [self.args.in_drop, self.args.in_drop, self.args.lr, self.args.weight_decay]
            instance = Individual(self.args, net_genes, param_genes)
            struct_individuals.append(instance)
        
        self.struct_individuals = struct_individuals

        param_genes = [self.args.in_drop, self.args.in_drop, self.args.lr, self.args.weight_decay]
        params_individuals = [param_genes]
        for j in range(self.args.num_individuals_param-1):
            param_genes = self.hybrid_search_space.get_param_instance()
            params_individuals.append(param_genes)
    
        self.params_individuals = params_individuals


#     def cal_superfitness(self, super_individual):
#         best_fitness = -1
#         average_fitness = 0
#         for elem in super_individual:
#             if best_fitness < elem.get_fitness():
#                 best_fitness = elem.get_fitness()
#             average_fitness += elem.get_fitness()
#         
#         average_fitness = average_fitness / self.args.num_individuals
#         final_fitness = self.args.super_ratio * best_fitness + (1-self.args.super_ratio) * average_fitness
#         
#         return final_fitness

    def init_param_population(self, init_individuals):
#         action = init_individuals.get_net_genes()
#         param = init_individuals.get_param_genes()
        super_population = []
        
        init_pop = Super_Individual(self.args, init_individuals, self.params_individuals[0])
        init_pop.cal_superfitness()
        super_population.append(init_pop)
        for i in range(self.args.num_individuals_param-1):
            individuals = copy.deepcopy(init_individuals)
#            param_genes = self.hybrid_search_space.get_param_instance()
            param_genes = self.params_individuals[i+1]
            for i in range(self.args.num_individuals): 
                individuals[i].param_genes = param_genes
                individuals[i].cal_fitness(self.gnn_manager)
            new_pop = Super_Individual(self.args, individuals, param_genes)
            new_pop.cal_superfitness()
            super_population.append(new_pop)
            
        self.super_population = super_population
                    
    
    # run on the single model with more training epochs
    def single_model_run(self, num_epochs, actions):
        self.args.epochs = num_epochs
        self.gnn_manager.train(actions)
        
        
    def cal_fitness(self):
        """calculate fitness scores of all individuals,
          e.g., the classification accuracy from GNN"""
        for individual in self.struct_individuals:
            individual.cal_fitness(self.gnn_manager)
            
    def parent_selection(self):
        "select k individuals by fitness probability"
        k = self.args.num_parents
        
        # select the parents for structure evolution
        fitnesses = [i.get_fitness() for i in self.struct_individuals]
        fit_probs = fitnesses / np.sum(fitnesses)
        struct_parents = choices(self.struct_individuals, k=k, weights=fit_probs)
        
        return struct_parents
    
    def parent_selection_param(self):
        "select k individuals by fitness probability"
        k = self.args.num_parents_param
        
        # select the parents for structure evolution
        fitnesses = [i.get_fitness() for i in self.super_population]
        fit_probs = fitnesses / np.sum(fitnesses)
        param_parents = choices(self.super_population, k=k, weights=fit_probs)
        
        return param_parents
    
    def crossover_net(self, parents):  
        "produce offspring from parents for better net architecture"
        p_size = len(parents)
        maximum = p_size * (p_size - 1) / 2
        if self.args.num_offsprings > maximum:
            raise RuntimeError("number of offsprings should not be more than " 
                               + maximum)
            
        # randomly choose crossover parent pairs
        parent_pairs = []
        while len(parent_pairs) < self.args.num_offsprings:
            indexes = sample(range(p_size), k=2)
            pair = (indexes[0], indexes[1])
            if indexes[0] > indexes[1]:
                pair = (indexes[1], indexes[0])
            if not pair in parent_pairs:
                parent_pairs.append(pair)
        
#         print(parent_pairs)
        # crossover to generate offsprings
        offsprings = []
        gene_size = len(parents[0].get_net_genes())
        for i, j in parent_pairs:
            parent_gene_i = parents[i].get_net_genes()
            parent_gene_j = parents[j].get_net_genes()
            # select a random crossover point
            point_index = parent_gene_j.index(sample(parent_gene_j, 1)[0])
            offspring_gene_i = parent_gene_j[:point_index]
            offspring_gene_i.extend(parent_gene_i[point_index:])
            offspring_gene_j = parent_gene_i[:point_index]
            offspring_gene_j.extend(parent_gene_j[point_index:])
            
            # create offspring individuals
            offspring_i = Individual(self.args, offspring_gene_i, 
                                     parents[i].get_param_genes())
            offspring_j = Individual(self.args, offspring_gene_j, 
                                     parents[j].get_param_genes())
            
            offsprings.append([offspring_i, offspring_j])
            
        return offsprings   

    def crossover_param(self, parents):
        p_size = len(parents)
        maximum = p_size * (p_size - 1) / 2
        if self.args.num_offsprings_param > maximum:
            raise RuntimeError("number of offsprings should not be more than " 
                               + maximum)
        # randomly choose crossover parent pairs
        parent_pairs = []
        while len(parent_pairs) < self.args.num_offsprings_param:
            indexes = sample(range(p_size), k=2)
            pair = (indexes[0], indexes[1])
            if indexes[0] > indexes[1]:
                pair = (indexes[1], indexes[0])
            if not pair in parent_pairs:
                parent_pairs.append(pair)

        offsprings = []
        params_size = len(parents[0].get_param_genes())
        for i, j in parent_pairs:
            parent_gene_i = parents[i].get_param_genes()
            parent_gene_j = parents[j].get_param_genes()
            # select a random crossover point
            point_index = parent_gene_j.index(sample(parent_gene_j, 1)[0])
            offspring_gene_i = parent_gene_j[:point_index]
            offspring_gene_i.extend(parent_gene_i[point_index:])
            offspring_gene_j = parent_gene_i[:point_index]
            offspring_gene_j.extend(parent_gene_j[point_index:])
            
            # create offspring individuals
            offspring_i = copy.deepcopy(parents[i])
            offspring_j = copy.deepcopy(parents[j])
            offspring_i.set_newparam(offspring_gene_i)
            offspring_j.set_newparam(offspring_gene_j)
            
            offsprings.append([offspring_i, offspring_j])
        return offsprings  
        
    
    def mutation_net(self, offsprings):
        """perform mutation for all new offspring individuals"""
        for pair in offsprings:
            random_prob = np.random.uniform(0, 1, 1)
            if random_prob <= self.args.mutate_prob:
                index, gene = self.hybrid_search_space.get_one_net_gene()
                pair[0].mutation_net_gene(index, gene, 'struct')
            random_prob = np.random.uniform(0, 1, 1)
            if random_prob <= self.args.mutate_prob:
                index, gene = self.hybrid_search_space.get_one_net_gene()
                pair[1].mutation_net_gene(index, gene, 'struct')
                
    def mutation_param(self, offsprings):
        """perform mutation for all new offspring individuals"""
        for pair in offsprings:
            random_prob = np.random.uniform(0, 1, 1)
            if random_prob <= self.args.mutate_prob:
                index, gene = self.hybrid_search_space.get_one_param_gene()
                pair[0].mutation_net_gene(index, gene, 'param')
            random_prob = np.random.uniform(0, 1, 1)
            if random_prob <= self.args.mutate_prob:
                index, gene = self.hybrid_search_space.get_one_param_gene()
                pair[1].mutation_net_gene(index, gene, 'param')
                
    def find_least_fittest(self, individuals):
        fitness = 10000
        index =-1
        for elem_index, elem in enumerate(individuals):
            if fitness > elem.get_fitness():
                fitness = elem.get_fitness()
                index = elem_index
                
        return index
                           
    def cal_fitness_offspring(self, offsprings):
        survivors = []
        for pair in offsprings:
            offspring_1 = pair[0]
            offspring_2 = pair[1]
            offspring_1.cal_fitness(self.gnn_manager)
            offspring_2.cal_fitness(self.gnn_manager)
            if offspring_1.get_fitness() > offspring_2.get_fitness():
                survivors.append(offspring_1)
            else:
                survivors.append(offspring_2)
        
        return survivors
    
    def cal_fitness_offspring_param(self, offsprings):
        survivors = []
        for pair in offsprings:
            offspring_1 = pair[0].get_population()
            offspring_2 = pair[1].get_population()
            for i in range(self.args.num_individuals):
                offspring_1[i].cal_fitness(self.gnn_manager)
            for j in range(self.args.num_individuals):
                offspring_2[i].cal_fitness(self.gnn_manager)
            pair[0].cal_superfitness()
            pair[1].cal_superfitness()
            
            if pair[0].get_fitness() > pair[1].get_fitness():
                survivors.append(pair[0])
            else:
                survivors.append(pair[1])
        
        return survivors
    
    def update_struct(self, elem):
        for i in range(self.args.num_individuals):
            individual = self.struct_individuals[i]
            if self.compare_action(individual.get_net_genes(), elem.get_net_genes()):
                if individual.get_fitness() < elem.get_fitness():
                    self.struct_individuals[i] = elem
                    return False
        return True
            
    def update_population_struct(self, survivors):
        """update current population with new offsprings"""
        for elem in survivors:
            if self.update_struct(elem):
                out_index = self.find_least_fittest(self.struct_individuals)
                self.struct_individuals[out_index] = elem

    def compare_action(self, a1, a2):
        for i in range(len(a1)):
            if a1[i] != a2[i]:
                return False
        return True

    def combine_population(self, param_genes):
        for elem in self.param_individuals:
            elem.param_genes

            
    def update_population_param(self, survivors):
        """update current population with new offsprings"""
        for elem in survivors:
            out_index = self.find_least_fittest(self.super_population)
            self.super_population[out_index] = elem
        params_individuals = []
        for super_elem in self.super_population:
            params_individuals.append(super_elem.get_param_genes())
        self.params_individuals = params_individuals

    
    def print_models(self, iter):
        
        print('===begin, current population ({} in {} generations)===='.format(
                                    (iter+1), self.args.num_generations))
        
        best_individual = self.struct_individuals[0]
        for elem_index, elem in enumerate(self.struct_individuals):
            if best_individual.get_fitness() < elem.get_fitness():
                best_individual = elem
            print('struct space: {}, param space: {}, validate_acc={}, test_acc={}'.format(
                                elem.get_net_genes(), 
                                elem.get_param_genes(),
                                elem.get_fitness(),
                                elem.get_test_acc()))
        print('------the best model-------')
        print('struct space: {}, param space: {}, validate_acc={}, test_acc={}'.format(
                           best_individual.get_net_genes(), 
                           best_individual.get_param_genes(),
                           best_individual.get_fitness(),
                           best_individual.get_test_acc()))    
           
        print('====end====\n')
        
        return best_individual
    
                    
    def evolve_net(self):
        # initialize population
        self.init_population()
        # calculate fitness for population
        self.cal_fitness()
        
        actions = []
        params = []
        train_accs = []
        test_accs = []
        
        for j in range(self.args.num_generations):
            # GNN hyper parameter evolution
            print('===================GNN hyper parameter evolution====================')
            initl_param_individual = copy.deepcopy(self.struct_individuals)
            self.init_param_population(initl_param_individual)

            for i in range(self.args.num_generations_param):
                param_parents = self.parent_selection_param()            
                param_offsprings = self.crossover_param(param_parents)
                self.mutation_param(param_offsprings)
                param_survivors = self.cal_fitness_offspring_param(param_offsprings)
                self.update_population_param(param_survivors) # update the population      
            
            # update the structure population with the best hyper-parameter
            print('##################update structure population##################')
            out_index = self.find_least_fittest(self.super_population)
            self.struct_individuals = self.super_population[out_index].get_population()
            
            # GNN structure evolution
            print('--------------------GNN structure evolution-------------------------')
            struct_parents = self.parent_selection() # parents selection
            struct_offsprings = self.crossover_net(struct_parents) # crossover to produce offsprings
            self.mutation_net(struct_offsprings) # perform mutation
            struct_survivors = self.cal_fitness_offspring(struct_offsprings) # calculate fitness for offsprings
            self.update_population_struct(struct_survivors) # update the population 
            
            
            best_individual = self.print_models(j)
            actions.append(best_individual.get_net_genes())
            params.append(best_individual.get_param_genes())
            train_accs.append(best_individual.get_fitness())
            test_accs.append(best_individual.get_test_acc())
        
            print(actions)           
            print(params)           
            print(train_accs)           
            print(test_accs)           
        
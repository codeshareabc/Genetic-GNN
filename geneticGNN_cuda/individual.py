import numpy as np

class Individual(object):
    
    def __init__(self, args, net_genes, param_genes):
        
        self.args = args
        self.net_genes = net_genes
        self.param_genes = param_genes

        
    def get_net_genes(self):
        return self.net_genes
    
    def get_param_genes(self):
        return self.param_genes
    
    def cal_fitness(self, gnn_manager):
        # run gnn to get the classification accuracy as fitness
        val_acc, test_acc = gnn_manager.train(self.net_genes, self.param_genes)
        self.fitness = val_acc
        self.test_acc = test_acc
        
    def get_fitness(self):
        return self.fitness
    
    def get_test_acc(self):
        return self.test_acc
    
    def mutation_net_gene(self, mutate_point, new_gene, type='struct'):
        if type == 'struct':
            self.net_genes[mutate_point] = new_gene
        elif type == 'param':
            self.param_genes[mutate_point] = new_gene
        else:
            raise Exception("wrong mutation type")
        
        

import torch
from random import sample

class HybridSearchSpace(object):
    def __init__(self, num_gnn_layers):
        
        self.num_gnn_layers = num_gnn_layers
        
        self.net_space = {
            'attention_type':['gat', 'gcn', 'cos', 'const', 'gat_sym', 'linear', 'generalized_linear'],
#             'aggregator_type':['sum', 'mean', 'max', 'mlp', 'lstm'],
            'aggregator_type':['sum', 'mean', 'max', 'mlp'],
            'activation_function':['sigmoid', 'tanh', 'relu', 'linear',
                                   'softplus', 'leaky_relu', 'relu6', 'elu'],
            'number_of_heads':[1, 2, 4, 6, 8, 16],
            'hidden_units':[4, 8, 16, 32, 64, 128, 256]
            }
            
        self.param_space = {
            'drop_out':[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'learning_rate':[5e-4, 1e-3, 5e-3, 1e-2],
            'weight_decay':[5e-4, 8e-4, 1e-3, 4e-3]
            }
    
    def get_net_space(self):
        return self.net_space
    
    def get_param_space(self):
        return self.param_space   
    
    def get_action_type_list(self):
        action_names = list(self.net_space.keys())
        return action_names * self.num_gnn_layers
    
    def get_net_instance(self):
        "sample network architects for multi-layer GNN"
        net_architects = []
        net_space = self.get_net_space()
        for i in range(self.num_gnn_layers):
            for action_name in net_space.keys():
                actions =  net_space[action_name]
                net_architects.extend(sample(actions, 1))
            
        return net_architects
    
    def get_param_instance(self):
        "sample network hyper parameters"
        net_parameters = []
        param_space = self.get_param_space()
        for i in range(self.num_gnn_layers):
            params =  param_space['drop_out']
            net_parameters.extend(sample(params, 1))

        params = param_space['learning_rate']
        net_parameters.extend(sample(params, 1))
        params = param_space['weight_decay']
        net_parameters.extend(sample(params, 1))                

        return net_parameters
        
        
    def get_one_net_gene(self):
        "randomly sample a gene for mutation from the architecture space"
        action_type_list = self.get_action_type_list()
        gene_mutate_index = sample(range(len(action_type_list)), 1)[0]
        gene_mutate_candidates = self.net_space[action_type_list[gene_mutate_index]]
        gene_mutate_to = sample(gene_mutate_candidates, 1)[0]
        
        return gene_mutate_index, gene_mutate_to
    
    def get_one_param_gene(self):
        "randomly sample a gene for mutation from the architecture space"
        param_len = self.num_gnn_layers + 2
        param_type_list = ['drop_out'] * self.num_gnn_layers + ['learning_rate', 'weight_decay']
        gene_mutate_index = sample(range(param_len), 1)[0]
        gene_mutate_candidates = self.param_space[param_type_list[gene_mutate_index]]
        gene_mutate_to = sample(gene_mutate_candidates, 1)[0]
        
        return gene_mutate_index, gene_mutate_to
            
        
        
        
        
def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return torch.nn.functional.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")
        
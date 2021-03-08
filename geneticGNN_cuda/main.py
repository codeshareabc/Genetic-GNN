import argparse
import time
import torch
import configs
import tensor_utils as utils
from population import Population

def main(args):
    torch.manual_seed(args.random_seed) 
    
    utils.makedirs(args.dataset)  
    
    print(args.super_ratio)
    
    pop = Population(args)
    pop.evolve_net()

#     # run on single model
#     num_epochs = 200
#     actions = ['gcn', 'mean', 'softplus', 16, 8, 'gcn', 'max', 'tanh', 16, 6] 
#     pop.single_model_run(num_epochs, actions)
    
    
if __name__ == "__main__":
    args = configs.build_args('GeneticGNN')
    main(args)
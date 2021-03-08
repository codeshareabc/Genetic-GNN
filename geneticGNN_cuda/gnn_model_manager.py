import numpy as np
import os.path as osp
import time
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from gnn import GraphNet

import warnings
warnings.filterwarnings('ignore')


def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()

class GNNModelManager(object):
    
    def __init__(self, args):
        
        self.args = args
        self.loss_fn = torch.nn.functional.nll_loss
        
    
    def load_data(self, dataset='Citeseer'):
        
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        dataset = Planetoid(path, dataset, T.NormalizeFeatures())
        data = dataset[0]
        
#         print(np.sum(np.array(data.val_mask), 0))
        
#         data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
#         data.train_mask[:-1000] = 1
#         data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
#         data.val_mask[data.num_nodes - 1000: data.num_nodes - 500] = 1
#         data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
#         data.test_mask[data.num_nodes - 500:] = 1
        
        self.data = data
        
        
#         print(data.edge_index)
#         print(data.edge_index.shape)
        
        self.args.num_class = data.y.max().item() + 1
        self.args.in_feats = self.data.num_features
    
    def load_param(self):
        # don't share param
        pass    

    def update_args(self, args):
        self.args = args

    def save_param(self, model, update_all=False):
        pass

    def shuffle_data(self, full_data=True):
        device = torch.device('cuda' if self.args.cuda else 'cpu')
        if full_data:
            self.data = fix_size_split(self.data, self.data.num_nodes - 1000, 500, 500)
        else:
            self.data = fix_size_split(self.data, 1000, 500, 500)
        self.data.to(device)
            
        
    def build_gnn(self, actions, drop_outs):
        
        model = GraphNet(self.args.num_gnn_layers,
                         actions, self.args.in_feats, self.args.num_class, 
                         drop_outs=drop_outs, multi_label=False,
                         batch_normal=False, residual=False)
        return model
        
    # train from scratch
    def evaluate(self, actions=None, format="two"):
        actions = process_action(actions, format, self.args)
        print("train action:", actions)

        # create model
        model = self.build_gnn(actions)

        if self.args.cuda:
            model.cuda()

        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        try:
            model, val_acc, test_acc = self.run_model(model, optimizer, self.loss_fn, self.data, self.epochs,
                                                      cuda=self.args.cuda, return_best=True,
                                                      half_stop_score=max(self.reward_manager.get_top_average() * 0.7,
                                                                          0.4))
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
                val_acc = 0
                test_acc = 0
            else:
                raise e
        return val_acc, test_acc
        
    # train from scratch
    def train(self, actions=None, params=None):
        # change the last gnn dimension to num_class
        actions[-1] = self.args.num_class
        print('==================================\ncurrent training actions={}, params={}'.format(actions, params))
        
        # create gnn model
        learning_rate = params[-2]
        weight_decay = params[-1]
        drop_outs = params[:-2]
        
        gnn_model = self.build_gnn(actions, drop_outs)
        if self.args.cuda:
            gnn_model.cuda()        
        # define optimizer
        optimizer = torch.optim.Adam(gnn_model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
        
        # run model to get accuracy
        model, val_acc, test_acc = self.run_model(gnn_model, 
                                        optimizer, 
                                        self.loss_fn, 
                                        self.data, 
                                        self.args.epochs,
                                        show_info=False)

        return val_acc, test_acc
        
    @staticmethod
    def run_model(model, optimizer, loss_fn, data, epochs, early_stop=5, 
                  return_best=False, cuda=True, need_early_stop=False, show_info=False):
#        device = torch.device('cuda' if self.args.cuda else 'cpu')
        dur = []
        begin_time = time.time()
        best_performance = 0
        min_val_loss = float("inf")
        min_train_loss = float("inf")
        model_val_acc = 0
        model_test_acc = 0
#         print("Number of train datas:", data.train_mask.sum())
        for epoch in range(1, epochs + 1):
            
            model.train()
#             print(data.edge_index.shape, data.x.shape, data.y.shape)
            # forward
            
            logits = model(data.x.cuda(), data.edge_index.cuda())
            logits = F.log_softmax(logits, 1)
            
            loss = loss_fn(logits[data.train_mask], data.y[data.train_mask].cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            # evaluate
            model.eval()
            logits = model(data.x.cuda(), data.edge_index.cuda())
            logits = F.log_softmax(logits, 1)
            
            train_acc = evaluate(logits.cuda(), data.y.cuda(), data.train_mask.cuda())
            val_acc = evaluate(logits.cuda(), data.y.cuda(), data.val_mask.cuda())
            test_acc = evaluate(logits.cuda(), data.y.cuda(), data.test_mask.cuda())

            loss = loss_fn(logits[data.val_mask].cuda(), data.y[data.val_mask].cuda())
            val_loss = loss.item()
            if val_loss < min_val_loss:  # and train_loss < min_train_loss
                min_val_loss = val_loss
                min_train_loss = train_loss
                model_val_acc = val_acc
                model_test_acc = test_acc
                if test_acc > best_performance:
                    best_performance = test_acc
            if show_info:
                time_used = time.time() - begin_time
                print(
                    "Epoch {:05d} | Loss {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f} | time {}".format(
                        epoch, loss.item(), train_acc, val_acc, test_acc, time_used))

#                 print("Each Epoch Cost Time: %f " % ((end_time - begin_time) / epoch))
        print("val_score:{:.4f}, test_score:{:.4f}".format(model_val_acc, model_test_acc), '\n')
        if return_best:
            return model, model_val_acc, best_performance
        else:
            return model, model_val_acc, model_test_acc
        

    @staticmethod
    def prepare_data(data, cuda=True):
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        mask = torch.ByteTensor(data.train_mask)
        test_mask = torch.ByteTensor(data.test_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        n_edges = data.graph.number_of_edges()
        # create DGL graph
        g = DGLGraph(data.graph)
        # add self loop
        g.add_edges(g.nodes(), g.nodes())
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0

        if cuda:
            features = features.cuda()
            labels = labels.cuda()
            norm = norm.cuda()
        g.ndata['norm'] = norm.unsqueeze(1)
        return features, g, labels, mask, val_mask, test_mask, n_edges        
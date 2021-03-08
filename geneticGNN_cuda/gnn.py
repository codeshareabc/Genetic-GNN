import torch
import torch.nn as nn
import torch.nn.functional as F
from search_space import act_map
from pyg_gnn_layer import GeoLayer

class GraphNet(torch.nn.Module):
    
    def __init__(self, layer_nums, actions, num_feat, num_label, drop_outs, multi_label=False, 
                 batch_normal=True, residual=False, state_num=5):
        
        super(GraphNet, self).__init__()     
        
        # args
        self.layer_nums = layer_nums     
        self.multi_label = multi_label
        self.num_feat = num_feat
        self.num_label = num_label
        self.dropouts = drop_outs
        self.residual = residual   
        self.batch_normal = batch_normal 
        
        # layer module
        self.build_model(actions, batch_normal, drop_outs, num_feat, num_label, state_num)
        
        
        
    def build_model(self, actions, batch_normal, drop_outs, num_feat, num_label, state_num):
        if self.residual:
            self.fcs = torch.nn.ModuleList()
        if self.batch_normal:
            self.bns = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        self.acts = []
        self.gates = torch.nn.ModuleList()
        self.build_hidden_layers(actions, batch_normal, drop_outs, self.layer_nums, num_feat, num_label, state_num)
        
    def build_hidden_layers(self, actions, batch_normal, drop_outs, layer_nums, num_feat, num_label, state_num=6):

        # build hidden layer
        for i in range(layer_nums):

            if i == 0:
                in_channels = num_feat
            else:
                in_channels = out_channels * head_num

            # extract layer information
            attention_type = actions[i * state_num + 0]
            aggregator_type = actions[i * state_num + 1]
            act = actions[i * state_num + 2]
            head_num = actions[i * state_num + 3]
            out_channels = actions[i * state_num + 4]
            concat = True
            if i == layer_nums - 1:
                concat = False
            if self.batch_normal:
                self.bns.append(torch.nn.BatchNorm1d(in_channels, momentum=0.5))
            self.layers.append(
                GeoLayer(in_channels, out_channels, head_num, concat, dropout=drop_outs[i],
                         att_type=attention_type, agg_type=aggregator_type, ))
            self.acts.append(act_map(act))
            if self.residual:
                if concat:
                    self.fcs.append(torch.nn.Linear(in_channels, out_channels * head_num))
                else:
                    self.fcs.append(torch.nn.Linear(in_channels, out_channels))
            
    def forward(self, x, edge_index_all):
        output = x
        if self.residual:
            for i, (act, layer, fc) in enumerate(zip(self.acts, self.layers, self.fcs)):
                output = F.dropout(output, p=layer.dropout, training=self.training)
                if self.batch_normal:
                    output = self.bns[i](output)

                output = act(layer(output, edge_index_all) + fc(output))
        else:
            for i, (act, layer) in enumerate(zip(self.acts, self.layers)):
                output = F.dropout(output, p=layer.dropout, training=self.training)
                if self.batch_normal:
                    output = self.bns[i](output)
                output = act(layer(output, edge_index_all))
        if not self.multi_label:
            output = F.log_softmax(output, dim=1)
        return output


    def __repr__(self):
        result_lines = ""
        for each in self.layers:
            result_lines += str(each)
        return result_lines

    @staticmethod
    def merge_param(old_param, new_param, update_all):
        for key in new_param:
            if update_all or key not in old_param:
                old_param[key] = new_param[key]
        return old_param

    def get_param_dict(self, old_param=None, update_all=True):
        if old_param is None:
            result = {}
        else:
            result = old_param
        for i in range(self.layer_nums):
            key = "layer_%d" % i
            new_param = self.layers[i].get_param_dict()
            if key in result:
                new_param = self.merge_param(result[key], new_param, update_all)
                result[key] = new_param
            else:
                result[key] = new_param
        if self.residual:
            for i, fc in enumerate(self.fcs):
                key = "layer_{i}_fc_{fc.weight.size(0)}_{fc.weight.size(1)}"
                result[key] = self.fcs[i]
        if self.batch_normal:
            for i, bn in enumerate(self.bns):
                key = "layer_{i}_fc_{bn.weight.size(0)}"
                result[key] = self.bns[i]
        return result

    # load parameters from parameter dict
    def load_param(self, param):
        if param is None:
            return

        for i in range(self.layer_nums):
            self.layers[i].load_param(param["layer_%d" % i])

        if self.residual:
            for i, fc in enumerate(self.fcs):
                key = "layer_{i}_fc_{fc.weight.size(0)}_{fc.weight.size(1)}"
                if key in param:
                    self.fcs[i] = param[key]
        if self.batch_normal:
            for i, bn in enumerate(self.bns):
                key = "layer_{i}_fc_{bn.weight.size(0)}"
                if key in param:
                    self.bns[i] = param[key]


def gat_message(edges):
    if 'norm' in edges.src:
        msg = edges.src['ft'] * edges.src['norm']
        return {'ft': edges.src['ft'], 'a2': edges.src['a2'], 'a1': edges.src['a1'], 'norm': msg}
    return {'ft': edges.src['ft'], 'a2': edges.src['a2'], 'a1': edges.src['a1']}
            
class NASLayer(nn.Module):
    def __init__(self, attention_type, aggregator_type, act, head_num, in_channels, out_channels=8, concat=True,
                 dropout=0.6, pooling_dim=128, residual=False, batch_normal=True):
        '''
        build one layer of GNN
        :param attention_type:
        :param aggregator_type:
        :param act: Activation function
        :param head_num: head num, in another word repeat time of current ops
        :param in_channels: input dimension
        :param out_channels: output dimension
        :param concat: concat output. get average when concat is False
        :param dropout: dropput for current layer
        :param pooling_dim: hidden layer dimension; set for pooling aggregator
        :param residual: whether current layer has  skip-connection
        :param batch_normal: whether current layer need batch_normal
        '''
        super(NASLayer, self).__init__()
        # print("NASLayer", in_channels, concat, residual)
        self.attention_type = attention_type
        self.aggregator_type = aggregator_type
        self.act = NASLayer.act_map(act)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = int(head_num)
        self.concat = concat
        self.dropout = dropout
        self.attention_dim = 1
        self.pooling_dim = pooling_dim

        self.batch_normal = batch_normal

        if attention_type in ['cos', 'generalized_linear']:
            self.attention_dim = 64
        self.bn = nn.BatchNorm1d(self.in_channels, momentum=0.5)
        self.prp = nn.ModuleList()
        self.red = nn.ModuleList()
        self.fnl = nn.ModuleList()
        self.agg = nn.ModuleList()
        for hid in range(self.num_heads):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.prp.append(AttentionPrepare(in_channels, out_channels, self.attention_dim, dropout))
            agg = NASLayer.aggregator_map(aggregator_type, out_channels, pooling_dim)
            self.agg.append(agg)
            self.red.append(NASLayer.attention_map(attention_type, dropout, agg, self.attention_dim))
            self.fnl.append(GATFinalize(hid, in_channels,
                                        out_channels, NASLayer.act_map(act), residual))


    @staticmethod
    def aggregator_map(aggregator_type, in_dim, pooling_dim):
        if aggregator_type == "sum":
            return SumAggregator()
        elif aggregator_type == "mean":
            return MeanPoolingAggregator(in_dim, pooling_dim)
        elif aggregator_type == "max":
            return MaxPoolingAggregator(in_dim, pooling_dim)
        elif aggregator_type == "mlp":
            return MLPAggregator(in_dim, pooling_dim)
        elif aggregator_type == "lstm":
            return LSTMAggregator(in_dim, pooling_dim)
        elif aggregator_type == "gru":
            return GRUAggregator(in_dim, pooling_dim)
        else:
            raise Exception("wrong aggregator type", aggregator_type)

    @staticmethod
    def attention_map(attention_type, attn_drop, aggregator, attention_dim):
        if attention_type == "gat":
            return GATReduce(attn_drop, aggregator)
        elif attention_type == "cos":
            return CosReduce(attn_drop, aggregator)
        elif attention_type in ["none", "const"]:
            return ConstReduce(attn_drop, aggregator)
        elif attention_type == "gat_sym":
            return GatSymmetryReduce(attn_drop, aggregator)
        elif attention_type == "linear":
            return LinearReduce(attn_drop, aggregator)
        elif attention_type == "bilinear":
            return CosReduce(attn_drop, aggregator)
        elif attention_type == "generalized_linear":
            return GeneralizedLinearReduce(attn_drop, attention_dim, aggregator)
        elif attention_type == "gcn":
            return GCNReduce(attn_drop, aggregator)
        else:
            raise Exception("wrong attention type")

    @staticmethod
    def act_map(act):
        if act == "linear":
            return lambda x: x
        elif act == "elu":
            return F.elu
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

    def get_param_dict(self):
        params = {}

        key = "%d_%d_%d_%s" % (self.in_channels, self.out_channels, self.num_heads, self.attention_type)
        prp_key = key + "_" + str(self.attention_dim) + "_prp"
        agg_key = key + "_" + str(self.pooling_dim) + "_" + self.aggregator_type
        fnl_key = key + "_fnl"
        bn_key = "%d_bn" % self.in_channels
        params[prp_key] = self.prp.state_dict()
        params[agg_key] = self.agg.state_dict()
        # params[key+"_"+self.attention_type] = self.red.state_dict()
        params[fnl_key] = self.fnl.state_dict()
        params[bn_key] = self.bn.state_dict()
        return params

    def load_param(self, param):
        key = "%d_%d_%d_%s" % (self.in_channels, self.out_channels, self.num_heads, self.attention_type)
        prp_key = key + "_" + str(self.attention_dim) + "_prp"
        agg_key = key + "_" + str(self.pooling_dim) + "_" + self.aggregator_type
        fnl_key = key + "_fnl"
        bn_key = "%d_bn" % self.in_channels
        if prp_key in param:
            self.prp.load_state_dict(param[prp_key])

        # red_key = key+"_"+self.attention_type
        if agg_key in param:
            self.agg.load_state_dict(param[agg_key])
            for i in range(self.num_heads):
                self.red[i].aggregator = self.agg[i]

        if fnl_key in param:
            self.fnl.load_state_dict(param[fnl_key])

        if bn_key in param:
            self.bn.load_state_dict(param[bn_key])

    @staticmethod
    def merge_param(old_param, new_param, update_all):
        for key in new_param:
            if update_all or key not in old_param:
                old_param[key] = new_param[key]
        return old_param

    def forward(self, features, g):
        if self.batch_normal:
            last = self.bn(features)
        else:
            last = features

        for hid in range(self.num_heads):
            i = hid
            # prepare
            g.ndata.update(self.prp[i](last))
            # message passing
            g.update_all(gat_message, self.red[i], self.fnl[i])
        # merge all the heads
        if not self.concat:
            output = g.pop_n_repr('head0')
            for hid in range(1, self.num_heads):
                output = torch.add(output, g.pop_n_repr('head%d' % hid))
            output = output / self.num_heads
        else:
            output = torch.cat(
                [g.pop_n_repr('head%d' % hid) for hid in range(self.num_heads)], dim=1)
        del last
        return output


class AttentionPrepare(nn.Module):
    '''
        Attention Prepare Layer
    '''

    def __init__(self, input_dim, hidden_dim, attention_dim, drop):
        super(AttentionPrepare, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim, bias=False)
        if drop:
            self.drop = nn.Dropout(drop)
        else:
            self.drop = 0
        self.attn_l = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.attn_r = nn.Linear(hidden_dim, attention_dim, bias=False)
        nn.init.xavier_normal_(self.fc.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_l.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_r.weight.data, gain=1.414)

    def forward(self, feats):
        h = feats
        if self.drop:
            h = self.drop(h)
        ft = self.fc(h)
        a1 = self.attn_l(ft)
        a2 = self.attn_r(ft)
        return {'h': h, 'ft': ft, 'a1': a1, 'a2': a2}


######################################################################
# Reduce, apply different attention action and execute aggregation
######################################################################
class GATReduce(nn.Module):
    def __init__(self, attn_drop, aggregator=None):
        super(GATReduce, self).__init__()
        if attn_drop:
            self.attn_drop = nn.Dropout(p=attn_drop)
        else:
            self.attn_drop = 0
        self.aggregator = aggregator

    def apply_agg(self, neighbor):
        if self.aggregator:
            return self.aggregator(neighbor)
        else:
            return torch.sum(neighbor, dim=1)

    def forward(self, nodes):
        a1 = torch.unsqueeze(nodes.data['a1'], 1)  # shape (B, 1, 1)
        a2 = nodes.mailbox['a2']  # shape (B, deg, 1)
        ft = nodes.mailbox['ft']  # shape (B, deg, D)
        # attention
        a = a1 + a2  # shape (B, deg, 1)
        a = a.sum(-1, keepdim=True)  # Just in case the dimension is not zero
        e = F.softmax(F.leaky_relu(a), dim=1)
        if self.attn_drop:
            e = self.attn_drop(e)
        return {'accum': self.apply_agg(e * ft)}  # shape (B, D)


class ConstReduce(GATReduce):
    '''
        Attention coefficient is 1
    '''

    def __init__(self, attn_drop, aggregator=None):
        super(ConstReduce, self).__init__(attn_drop, aggregator)

    def forward(self, nodes):
        ft = nodes.mailbox['ft']  # shape (B, deg, D)
        # attention
        if self.attn_drop:
            ft = self.attn_drop(ft)
        return {'accum': self.apply_agg(1 * ft)}  # shape (B, D)


class GCNReduce(GATReduce):
    '''
        Attention coefficient is 1
    '''

    def __init__(self, attn_drop, aggregator=None):
        super(GCNReduce, self).__init__(attn_drop, aggregator)

    def forward(self, nodes):
        if 'norm' not in nodes.data:
            raise Exception("Wrong Data, has no norm")
        self_norm = nodes.data['norm']
        self_norm = self_norm.unsqueeze(1)
        results = nodes.mailbox['norm'] * self_norm
        return {'accum': self.apply_agg(results)}  # shape (B, D)


class LinearReduce(GATReduce):
    '''
        equal to neighbor's self-attention
    '''

    def __init__(self, attn_drop, aggregator=None):
        super(LinearReduce, self).__init__(attn_drop, aggregator)

    def forward(self, nodes):
        ft = nodes.mailbox['ft']  # shape (B, deg, D)
        a2 = nodes.mailbox['a2']
        a2 = a2.sum(-1, keepdim=True)  # shape (B, deg, D)
        # attention
        e = F.softmax(torch.tanh(a2), dim=1)
        if self.attn_drop:
            e = self.attn_drop(e)
        return {'accum': self.apply_agg(e * ft)}  # shape (B, D)


class GatSymmetryReduce(GATReduce):
    '''
        gat Symmetry version ( Symmetry cannot be guaranteed after softmax)
    '''

    def __init__(self, attn_drop, aggregator=None):
        super(GatSymmetryReduce, self).__init__(attn_drop, aggregator)

    def forward(self, nodes):
        a1 = torch.unsqueeze(nodes.data['a1'], 1)  # shape (B, 1, 1)
        b1 = torch.unsqueeze(nodes.data['a2'], 1)  # shape (B, 1, 1)
        a2 = nodes.mailbox['a2']  # shape (B, deg, 1)
        b2 = nodes.mailbox['a1']  # shape (B, deg, 1)
        ft = nodes.mailbox['ft']  # shape (B, deg, D)
        # attention
        a = a1 + a2  # shape (B, deg, 1)
        b = b1 + b2  # different attention_weight
        a = a + b
        a = a.sum(-1, keepdim=True)  # Just in case the dimension is not zero
        e = F.softmax(F.leaky_relu(a + b), dim=1)
        if self.attn_drop:
            e = self.attn_drop(e)
        return {'accum': self.apply_agg(e * ft)}  # shape (B, D)


class CosReduce(GATReduce):
    '''
        used in Gaan
    '''

    def __init__(self, attn_drop, aggregator=None):
        super(CosReduce, self).__init__(attn_drop, aggregator)

    def forward(self, nodes):
        a1 = torch.unsqueeze(nodes.data['a1'], 1)  # shape (B, 1, 1)
        a2 = nodes.mailbox['a2']  # shape (B, deg, 1)
        ft = nodes.mailbox['ft']  # shape (B, deg, D)
        # attention
        a = a1 * a2
        a = a.sum(-1, keepdim=True)  # shape (B, deg, 1)
        e = F.softmax(F.leaky_relu(a), dim=1)
        if self.attn_drop:
            e = self.attn_drop(e)
        return {'accum': self.apply_agg(e * ft)}  # shape (B, D)


class GeneralizedLinearReduce(GATReduce):
    '''
        used in GeniePath
    '''

    def __init__(self, attn_drop, hidden_dim, aggregator=None):
        super(GeneralizedLinearReduce, self).__init__(attn_drop, aggregator)
        self.generalized_linear = nn.Linear(hidden_dim, 1, bias=False)
        if attn_drop:
            self.attn_drop = nn.Dropout(p=attn_drop)
        else:
            self.attn_drop = 0

    def forward(self, nodes):
        a1 = torch.unsqueeze(nodes.data['a1'], 1)  # shape (B, 1, 1)
        a2 = nodes.mailbox['a2']  # shape (B, deg, 1)
        ft = nodes.mailbox['ft']  # shape (B, deg, D)
        # attention
        a = a1 + a2
        a = torch.tanh(a)
        a = self.generalized_linear(a)
        e = F.softmax(a, dim=1)
        if self.attn_drop:
            e = self.attn_drop(e)
        return {'accum': self.apply_agg(e * ft)}  # shape (B, D)


#######################################################
# Aggregators, aggregate information from neighbor
#######################################################

class SumAggregator(nn.Module):

    def __init__(self):
        super(SumAggregator, self).__init__()

    def forward(self, neighbor):
        return torch.sum(neighbor, dim=1)


class MaxPoolingAggregator(SumAggregator):

    def __init__(self, input_dim, pooling_dim=512, num_fc=1, act=F.leaky_relu_):
        super(MaxPoolingAggregator, self).__init__()
        out_dim = input_dim
        self.fc = nn.ModuleList()
        self.act = act
        if num_fc > 0:
            for i in range(num_fc - 1):
                self.fc.append(nn.Linear(out_dim, pooling_dim))
                out_dim = pooling_dim
            self.fc.append(nn.Linear(out_dim, input_dim))

    def forward(self, ft):
        for layer in self.fc:
            ft = self.act(layer(ft))

        return torch.max(ft, dim=1)[0]


class MeanPoolingAggregator(MaxPoolingAggregator):

    def __init__(self, input_dim, pooling_dim=512, num_fc=1, act=F.leaky_relu_):
        super(MeanPoolingAggregator, self).__init__(input_dim, pooling_dim, num_fc, act)

    def forward(self, ft):
        for layer in self.fc:
            ft = self.act(layer(ft))

        return torch.mean(ft, dim=1)


class MLPAggregator(MaxPoolingAggregator):

    def __init__(self, input_dim, pooling_dim=512, num_fc=1, act=F.leaky_relu_):
        super(MLPAggregator, self).__init__(input_dim, pooling_dim, num_fc, act)

    def forward(self, ft):
        ft = torch.sum(ft, dim=1)
        for layer in self.fc:
            ft = self.act(layer(ft))
        return ft


class LSTMAggregator(SumAggregator):

    def __init__(self, input_dim, pooling_dim=512):
        super(LSTMAggregator, self).__init__()
        self.lstm = nn.LSTM(input_dim, pooling_dim, batch_first=True, bias=False)
        self.linear = nn.Linear(pooling_dim, input_dim)

    def forward(self, ft):
        torch.transpose(ft, 1, 0)
        hidden = self.lstm(ft)[0]
        return self.linear(torch.squeeze(hidden[-1], dim=0))


class GRUAggregator(SumAggregator):

    def __init__(self, input_dim, pooling_dim=512):
        super(LSTMAggregator, self).__init__()
        self.lstm = nn.GRU(input_dim, pooling_dim, batch_first=True, bias=False)
        self.linear = nn.Linear(pooling_dim, input_dim)

    def forward(self, ft):
        torch.transpose(ft, 1, 0)
        hidden = self.lstm(ft)[0]
        return self.linear(torch.squeeze(hidden[-1], dim=0))


######################################################################
# Finalize, introduce residual connection
######################################################################
class GATFinalize(nn.Module):
    '''
        concat + fully connected layer
    '''

    def __init__(self, headid, indim, hiddendim, activation, residual):
        super(GATFinalize, self).__init__()
        self.headid = headid
        self.activation = activation
        self.residual = residual
        self.residual_fc = None
        if residual:
            if indim != hiddendim:
                self.residual_fc = nn.Linear(indim, hiddendim, bias=False)
                nn.init.xavier_normal_(self.residual_fc.weight.data, gain=1.414)

    def forward(self, nodes):
        ret = nodes.data['accum']
        if self.residual:
            if self.residual_fc is not None:
                ret = self.residual_fc(nodes.data['h']) + ret
            else:
                ret = nodes.data['h'] + ret
        return {'head%d' % self.headid: self.activation(ret)}
        
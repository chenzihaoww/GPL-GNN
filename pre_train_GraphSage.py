import os
import time
import warnings

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv

from config import get_args
from utils.utils import *

warnings.filterwarnings("ignore")

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        self.fc = nn.Linear(n_hidden, n_classes)

    def model_to_array(self, args):
        s_dict = torch.load('./data/' + args.dataset + '_model_' + args.file_id + '.pt')
        keys = list(s_dict.keys())
        res = s_dict[keys[0]].view(-1)
        for i in np.arange(1, len(keys), 1):
            res = torch.cat((res, s_dict[keys[i]].view(-1)))
        return res

    def array_to_model(self, arr):
        arr = self.model_to_array(args)
        m_m = torch.load('./data/' + args.dataset + '_model_' + args.file_id + '.pt')
        indice = 0
        s_dict = self.state_dict()
        for name, param in m_m.items():  # s_dict.items():
            length = torch.prod(torch.tensor(param.shape))
            s_dict[name] = arr[indice:indice + length].view(param.shape)
            indice = indice + length
        self.load_state_dict(s_dict)

    def load_parameters(self, args):
        self.args = args
        self.array_to_model(args)

    def forward(self, graph, output_nodes=None, Train_flag=False):
        inputs = graph[0].srcdata['feat']
        if not self.dropout:
            h = inputs
        else:
            h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h_dst = h[:graph[l].num_dst_nodes()]  # <---
            h = layer(graph[l], (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                if self.dropout:
                    h = self.dropout(h)
        h = self.activation(h)
        h = self.fc(h)
        return h
def save_res(acc_all, args, num=0):
    data = pd.DataFrame(acc_all)
    file_name = ''
    if args.half and args.mask_rate == 0.3:
        file_name = 'few_shot30'
    elif args.half and args.mask_rate == 0.5:
        file_name = 'few_shot50'
    elif not args.half:
        file_name = 'full'
    print(file_name, ',', num)
    if not os.path.exists(f'./res/gs_pre/{file_name}/' + args.dataset):
        os.makedirs(f'./res/gs_pre/{file_name}/' + args.dataset + '/')
    data.to_csv(f'./res/gs_pre/{file_name}/' + args.dataset + f'/self_{num}.csv', index=True,header=None)

def main(args,num=0):
    seed_torch(args.seed)
    g, features, labels, in_feats, n_classes, n_edges, train_nid, val_nid, test_nid, device = utils.get_init_info(args)
    sampler = dgl.dataloading.MultiLayerNeighborSampler(args.sample_list)
    train_dataloader = dgl.dataloading.NodeDataLoader(g, train_nid, sampler, device=device,
                                                      batch_size=args.batch_size, shuffle=True, drop_last=False,
                                                      num_workers=0)
    model = GraphSAGE(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.aggregator_type)
    model.to(device)
    model.load_parameters(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    acc_all = []
    loss_all = []
    for epoch in range(args.n_epochs):
        model.train()
        acc = evaluate(model, g, test_nid, args.batch_size, device, args.sample_list)
        acc_all.append(acc)
        t0 = time.time()
        for step, (input_nodes, output_nodes, mfgs) in enumerate(train_dataloader):
            inputs = mfgs[0].srcdata['feat']
            lab = mfgs[-1].dstdata['label']
            logits = model(mfgs, inputs)
            loss = F.cross_entropy(logits, lab)
            loss_all.append(loss.cpu().data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch {:03d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, time.time() - t0,
                                                                                      loss.item(), acc))

    if args.save_flag:
        save_res(acc_all, args,num)
    # acc = utils.evaluate(model, g, test_nid, args.batch_size, device, args.sample_list)
    # print("Test Accuracy {:.4f}".format(np.mean(acc_all[-10:])))


if __name__ == '__main__':
    args = get_args.get_my_args()
    for data in args.datasets_list:
        args.dataset = data
        for i in range(10):
            main(args, i)









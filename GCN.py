import random

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import get_args
from utils.utils import seed_torch, get_init_info, save_res
from dgl.nn.pytorch.conv import  GraphConv
import os


class Model(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, activation, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        # F.elu
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        self.layers.append(GraphConv(n_hidden, n_hidden))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_hidden, n_classes, bias=True)

    def reset_parameters(self):
        # for emb in self.layers:
        #     nn.init.xavier_uniform_(emb.weight.data)
        num_layers = len(self.layers)
        for i in range(num_layers):
            self.layers[i].reset_parameters()
        self.fc.reset_parameters()

    def forward(self, g, inputs):
        h = inputs
        for l, layer in enumerate(self.layers):
            if l != len(self.layers) - 1:
                h = self.dropout(h)
            h = layer(g, h)

        out = self.fc(h)
        return out


def evaluate_full_map(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train_full_map(g, features, labels, train_mask, test_mask, model, optimizer, args,num=0):
    loss_fcn = nn.CrossEntropyLoss()
    acc_all = []
    loss_all=[]
    max_acc = 0.0
    # training loop
    name = os.path.basename(__file__).split(".")[0]
    model_path = f'temp/model_{args.dataset}_{name}.pkl'
    for epoch in range(args.n_epochs):
        acc = evaluate_full_map(g, features, labels, test_mask, model)
        if acc > max_acc:
            torch.save(model.state_dict(), model_path)
            max_acc = acc
        acc_all.append(acc)
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )
        loss_all.append(loss.item())
    model.reset_parameters()
    model.load_state_dict(torch.load(model_path))
    acc = evaluate_full_map(g, features, labels, test_mask, model)
    print(f'val load para when max_cc={max_acc}')
    print(f'test :{acc}')
    if args.save_flag:
        save_res(acc_all, num, args, 'GCN-few')
        # save_res(loss_all, num, args, 'GCN-loss_256')
    # print('over')


def main_func(args, num=0):
    g, features, labels, in_feats, n_classes, n_edges, train_nid, val_nid, test_nid, device = get_init_info(args)
    g = dgl.add_self_loop(g)
    # create GAT model
    model = Model(in_feats, args.n_hidden, n_classes, F.relu, args.dropout).to(device)
    model.to(device)

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    model.train()
    acc_all = []
    max_acc = 0.0
    train_full_map(g.to(device), features.to(device), labels.to(device), train_nid, test_nid, model, optimizer, args,num)
    # print(f'val load para when max_cc={max_acc}')
    # print(f'test :{acc}')


if __name__ == '__main__':
    arg = get_args.get_my_args()
    datasets = ['cora', 'citeseer', 'pubmed', 'CoraFull', 'AmazonCoBuyComputer', 'AmazonCoBuyPhoto', 'ogbn-arxiv',
                'reddit']
    for data in datasets[:-1]:
        arg.dataset = data
        arg.seed = random.randint(3, 200)
        seed_torch(arg.seed)
        print(arg)
        print(arg.dataset)
        for i in range(0, 10):
            print('第', i, '次')
            main_func(arg, i)

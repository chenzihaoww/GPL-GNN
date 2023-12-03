import random

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import get_args
from utils.utils import seed_torch, get_init_info, save_res
from dgl.nn.pytorch.conv import GATConv
import os
from model import SAGE as Encoder


class GAT(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, heads, activation, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        # F.elu
        self.layers.append(GATConv(in_feats, n_hidden, heads[0],
                                   feat_drop=dropout,
                                   attn_drop=dropout, activation=activation))

        self.layers.append(
            GATConv(
                n_hidden * heads[0],
                n_hidden,
                heads[1],
                feat_drop=dropout,
                attn_drop=dropout,
                activation=activation,
            )
        )
        self.fc = nn.Linear(2 * n_hidden, n_classes, bias=True)

    def reset_parameters(self):
        # for emb in self.layers:
        #     nn.init.xavier_uniform_(emb.weight.data)
        num_layers = len(self.layers)
        for i in range(num_layers):
            self.layers[i].reset_parameters()
        self.fc.reset_parameters()

    def forward(self, g, inputs, h_dst=None):
        h = inputs
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != len(self.layers) - 1:
                h = h.flatten(1)
            else:  # last layer
                h = h.mean(1)
        h = torch.cat((h, h_dst), dim=1)
        out = self.fc(h)
        return out


def evaluate_full_map(g, features, labels, mask, model, emb):
    model.eval()
    with torch.no_grad():
        logits = model(g, features, emb)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train_full_map(g, features, labels, train_mask, test_mask, model, optimizer, args, num=0, emb=None):
    loss_fcn = nn.CrossEntropyLoss()
    acc_all = []
    max_acc = 0.0
    # training loop
    name = os.path.basename(__file__).split(".")[0]
    model_path = f'temp/model_{args.dataset}_{name}.pkl'
    for epoch in range(args.n_epochs):
        acc = evaluate_full_map(g, features, labels, test_mask, model, emb)
        if acc > max_acc:
            torch.save(model.state_dict(), model_path)
            max_acc = acc
        acc_all.append(acc)
        model.train()
        logits = model(g, features, emb)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )
    model.reset_parameters()
    model.load_state_dict(torch.load(model_path))
    acc = evaluate_full_map(g, features, labels, test_mask, model, emb)
    print(f'val load para when max_cc={max_acc}')
    print(f'test :{acc}')
    if args.save_flag:
        save_res(acc_all, num, args, 'down-GAT-few')
    # print('over')


def main_func(args, num=0):
    g, features, labels, in_feats, n_classes, n_edges, train_nid, val_nid, test_nid, device = get_init_info(args)
    g = dgl.add_self_loop(g)
    # create GAT model
    model = GAT(in_feats, args.n_hidden, n_classes, [8, 1], F.relu, args.dropout).to(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    encoder = Encoder(in_feats, args.n_hidden, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout)
    m_m = torch.load('./pre_data/' + args.dataset + '_model_' + args.file_id + '.pt', map_location=device)
    encoder.load_state_dict(m_m)
    encoder.eval()
    encoder.to(device)
    with torch.no_grad():
        emb = encoder.inference(g, features, device, 4000, 0)
    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)
    train_full_map(g.to(device), features.to(device), labels.to(device), train_nid, test_nid, model, optimizer, args,
                   num, emb.to(device))


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

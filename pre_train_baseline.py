import os.path
import os.path
import time
import warnings

# import matplotlib.pyplot as plt
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from config import get_args
from utils.utils import seed_torch, get_init_info,save_res
import random
warnings.filterwarnings("ignore")

from model import SAGE as Encoder


class FC(nn.Module):
    def __init__(self,
                 n_hidden,
                 n_classes
                 ):
        super().__init__()
        # self.device = device
        # self.layers = nn.ModuleList()
        # self.dropout = nn.Dropout(dropout)
        # self.activation = activation
        # self.n_classes = n_classes
        # self.center_num = center_num
        # self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # # hidden layers
        # for i in range(n_layers - 1):
        #     self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        self.FC = nn.Linear(n_hidden, n_classes, bias=True)

    def forward(self, h):
        out = self.FC(h)
        return out





def evaluate(model, x, y):
    pred = model(x)
    pred = pred.argmax(1)
    acc = sklearn.metrics.accuracy_score(y.cpu(), pred.cpu())
    f_score=sklearn.metrics.f1_score(y.cpu(), pred.cpu(),average='weighted')
    return acc,f_score


def main_func(args, num):
    g, features, labels, in_feats, n_classes, n_edges, train_nid, val_nid, test_nid, device = get_init_info(args)
    model=FC(args.n_hidden, n_classes)
    # model = GraphSAGE(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.aggregator_type,
    #                   args.center_num, device)
    model.to(device)
    # model.load_parameters(args)
    encoder = Encoder(in_feats, args.n_hidden, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout)
    m_m = torch.load('./pre_data/' + args.dataset + '_model_' + args.file_id + '.pt', map_location=device)
    encoder.load_state_dict(m_m)
    encoder.eval()
    encoder.to(device)
    with torch.no_grad():
        emb = encoder.inference(g, features, device, 4000, 0)
    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)
    # g.ndata['gfeat'] = emb

    # lr = sklearn.linear_model.LogisticRegression(multi_class='multinomial', max_iter=10000)
    # lr.fit(emb[train_nid].cpu(), labels[train_nid].cpu())
    # pred = lr.predict(emb)
    # f1_micro_eval = sklearn.metrics.f1_score(labels[test_nid].cpu(), pred[test_nid], average='micro')
    # acc_eval = sklearn.metrics.accuracy_score(labels[test_nid].cpu(), pred[test_nid])
    # print('f1_micro_eval:', f1_micro_eval)
    # print('acc_eval:', acc_eval)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_ds = TensorDataset(emb[train_nid], labels[train_nid])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    acc_all = []
    f_score_all=[]
    # loss_all = []
    max_acc = 0.0
    name = os.path.basename(__file__).split(".")[0]
    model_path = f'temp/model_{args.dataset}_{name}.pkl'
    for epoch in range(args.n_epochs):
        acc, f_score  = evaluate(model, emb[test_nid].to(device), labels[test_nid].to(device))
        if acc > max_acc:
            torch.save(model.state_dict(), model_path)
            max_acc = acc
        acc_all.append(acc)
        f_score_all.append(f_score)
        model.train()
        t0 = time.time()
        loss = None
        for step, (x, y) in enumerate(train_dl):
            optimizer.zero_grad()
            pred = model(x.to(device))
            loss = F.cross_entropy(pred, y.to(device))
            loss.backward()
            optimizer.step()
        print("Epoch {:03d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f}|f1_score {:.4f} ".format(epoch,
                                                                                                      time.time() - t0,
                                                                                                      loss.item(), acc,

                                                                                                      f_score))


    print(args.dataset)
    print(f'val load para when max_cc={max_acc}')
    if args.save_flag:
        save_res(acc_all, num, args,'pretrain')
    print('over')


if __name__ == '__main__':
    arg = get_args.get_my_args()
    seed_torch(arg.seed)
    # datasets=['cora','citeseer','pubmed','CoraFull','AmazonCoBuyComputer','AmazonCoBuyPhoto','ogbn-arxiv','reddit']
    datasets = ['cora']
    for data in datasets:
        arg.dataset = data
        arg.seed = random.randint(3, 200)
        seed_torch(arg.seed)
        print(arg)
        print(f'half:{arg.half},mask_rate:{arg.mask_rate}')
        print(arg.dataset)
        for i in range(0, 1):
            print('第', i, '次')
            main_func(arg, i)

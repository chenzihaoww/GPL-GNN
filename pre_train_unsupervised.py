import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.multiprocessing as mp
import dgl.function as fn
import time
import argparse
from torch.nn.parallel import DistributedDataParallel
import utils

from model import SAGE, compute_acc_unsupervised as compute_acc
from config import get_args

class NegativeSampler(object):
    def __init__(self, g, k, neg_share=False):
        self.weights = g.in_degrees().float() ** 0.75
        self.k = k
        self.neg_share = neg_share

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)
        n = len(src)
        if self.neg_share and n % self.k == 0:
            dst = self.weights.multinomial(n, replacement=True)
            dst = dst.view(-1, 1, self.k).expand(-1, self.k, -1).flatten()
        else:
            dst = self.weights.multinomial(n * self.k, replacement=True)
        src = src.repeat_interleave(self.k)
        return src, dst


class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        score = th.cat([pos_score, neg_score])
        label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)]).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss


def evaluate(model, g, nfeat, labels, train_nids, val_nids, test_nids, device):
    model.eval()
    with th.no_grad():
        # single gpu
        if isinstance(model, SAGE):
            pred = model.inference(g, nfeat, device, args.batch_size, args.num_workers)
        # multi gpu
        else:
            pred = model.module.inference(g, nfeat, device, args.batch_size, args.num_workers)
    model.train()

    return compute_acc(pred, labels, train_nids, val_nids, test_nids)


def evaluate2(model, g, nfeat, labels, train_nids, val_nids, test_nids, device):
    model.eval()
    with th.no_grad():
        # single gpu
        if isinstance(model, SAGE):
            pred = model.inference(g, nfeat, device, args.batch_size, args.num_workers)
        # multi gpu
        else:
            pred = model.module.inference(g, nfeat, device, args.batch_size, args.num_workers)

    return pred


#### Entry point
def run(proc_id, n_gpus, args, devices, data):
    # Unpack data
    device = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)
    train_mask, val_mask, test_mask, n_classes, g = data
    g = dgl.add_self_loop(g)
    nfeat = g.ndata.pop('feat')
    labels = g.ndata.pop('label')
    if not args.data_cpu:
        nfeat = nfeat.to(device)
        labels = labels.to(device)
    in_feats = nfeat.shape[1]
    train_nid = th.LongTensor(np.nonzero(train_mask)).squeeze()
    val_nid = th.LongTensor(np.nonzero(val_mask)).squeeze()
    test_nid = th.LongTensor(np.nonzero(test_mask)).squeeze()

    # Create PyTorch DataLoader for constructing blocks
    n_edges = g.num_edges()
    train_seeds = th.arange(n_edges)

    if args.sample_gpu:
        assert n_gpus > 0, "Must have GPUs to enable GPU sampling"
        train_seeds = train_seeds.to(device)
        g = g.to(device)

    # Create sampler
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_seeds, sampler, exclude='reverse_id',
        # For each edge with ID e in Reddit dataset, the reverse edge is e ± |E|/2.
        reverse_eids=th.cat([
            th.arange(n_edges // 2, n_edges),
            th.arange(0, n_edges // 2)]).to(train_seeds),
        negative_sampler=NegativeSampler(g, args.num_negs, args.neg_share),
        device=device,
        use_ddp=n_gpus > 1,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, args.num_hidden, len(set(labels)), args.num_layers, F.relu, args.dropout)
    # print(model)
    model.to(device)
    if n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)
    loss_fcn = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Training loop
    avg = 0
    iter_pos = []
    iter_neg = []
    iter_d = []
    iter_t = []
    best_eval_acc = 0
    best_test_acc = 0

    for epoch in range(args.num_epochs):
        if n_gpus > 1:
            dataloader.set_epoch(epoch)
        tic = time.time()

        tic_step = time.time()
        for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
            batch_inputs = nfeat[input_nodes].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            blocks = [block.int().to(device) for block in blocks]
            d_step = time.time()

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, pos_graph, neg_graph)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t = time.time()
            pos_edges = pos_graph.num_edges()
            neg_edges = neg_graph.num_edges()
            iter_pos.append(pos_edges / (t - tic_step))
            iter_neg.append(neg_edges / (t - tic_step))
            iter_d.append(d_step - tic_step)
            iter_t.append(t - d_step)
            if step % args.log_every == 0:
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print(
                    'proc_id:[{}],Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.4f}|{:.4f} | Load {:.4f}| train {:.4f} | GPU {:.1f} MB'.format(
                        proc_id, epoch, step, loss.item(), np.mean(iter_pos[3:]), np.mean(iter_neg[3:]),
                        np.mean(iter_d[3:]), np.mean(iter_t[3:]), gpu_mem_alloc))
            tic_step = time.time()


            if step % args.eval_every == 0 and proc_id == 0:
                eval_acc, test_acc = evaluate(model, g, nfeat, labels, train_nid, val_nid, test_nid,
                                              device)
                print('Eval Acc {:.4f} Test Acc {:.4f}'.format(eval_acc, test_acc))
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    best_test_acc = test_acc
                print('Best Eval Acc {:.4f} Test Acc {:.4f}'.format(best_eval_acc, best_test_acc))
                # gc.collect()
                # torch.cuda.empty_cache()
        toc = time.time()
        if proc_id == 0:
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if n_gpus > 1:
            th.distributed.barrier()
    # th.save(model.state_dict(), './data/' + args.dataset + '_model_' + args.file_id + '.pt')
    # m_state_dict = torch.load('./data/' + args.dataset + '_model_' + args.file_id + '.pt')  ####
    # model.load_state_dict(m_state_dict)  ####
    # # print("aaa")
    # pre = evaluate2(model, g, nfeat, labels, train_nid, val_nid, test_nid, device)
    # res = pre.detach().clone().cpu().data.numpy()
    # res = pd.DataFrame(res)
    # res.to_csv('./data/' + args.dataset + '_feat_' + args.file_id + '.csv', header=None, index=None)
    # # print(res.shape)
    # # print("bbbb")
    # print("##########\n", compute_acc(pre.detach().clone(), labels, train_nid, val_nid, test_nid))
    # if proc_id == 0:
    #     print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    # return model


def main(args, devices):
    utils.seed_torch(args.seed)
    g, features, labels, train_mask, val_mask, test_mask, in_feats, n_classes, n_edges = utils.my_load_data(args)

    # This avoids creating certain formats in each sub-process, which saves memory and CPU.
    g.create_formats_()
    # Pack data
    data = train_mask, val_mask, test_mask, n_classes, g
    n_gpus = len(devices)
    if devices[0] == -1:
        run(0, 0, args, ['cpu'], data)
    elif n_gpus == 1:
        run(0, n_gpus, args, devices, data)
    else:
        procs = []
        for proc_id in range(n_gpus):
            p = mp.Process(target=run, args=(proc_id, n_gpus, args, devices, data))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()


if __name__ == '__main__':
    a = get_args.get_my_args()
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument("--gpu", type=str, default='0', help="GPU, can be a list of gpus for multi-gpu training,"
                                                                " e.g., 0,1,2,3; -1 for CPU")
    argparser.add_argument('--dataset', type=str, default=a.dataset)
    argparser.add_argument('--dataset_dir', type=str, default=a.dataset_dir)
    argparser.add_argument('--num-epochs', type=int, default=a.n_epochs)
    argparser.add_argument('--num-hidden', type=int, default=a.n_hidden)
    argparser.add_argument('--num-layers', type=int, default=a.n_layers)
    argparser.add_argument('--num-negs', type=int, default=2)
    argparser.add_argument("--seed", type=int, default=200, help="random seed")
    argparser.add_argument("--aggregator-type", type=str, default='none', help="Aggregator type: mean/gcn/pool/lstm")
    argparser.add_argument('--neg-share', default=False, action='store_true',
                           help="sharing neg nodes for positive nodes")
    argparser.add_argument("--file-id", type=str, default='128GCN',  # reddit 是16
                           help="file id means feature shape")
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=4096)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=2000)
    argparser.add_argument('--lr', type=float, default=2e-4)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--half', type=bool, default=False)
    argparser.add_argument('--mask_rate', type=float, default=0)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--sample-gpu', action='store_true',
                           help="Perform the sampling process on the GPU. Must have 0 workers.")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    args = argparser.parse_args()
    devices = list(map(int, args.gpu.split(',')))
    # print(devices, type(devices))
    main(args, devices)

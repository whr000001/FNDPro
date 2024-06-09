import json
import os

import torch
from dataset import get_data
from torch_geometric.loader import NeighborLoader
from model import TaHid
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import os.path as osp
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--mode', type=str)
args = parser.parse_args()
dataset_name = args.dataset
mode = args.mode
assert dataset_name in ['politifact', 'gossipcop', 'mixed']
assert mode in ['RGCN', 'GCN', 'GAT', 'HGT']

save_path = 'checkpoints_{}_{}'.format(dataset_name, mode)
if not osp.exists(save_path):
    os.makedirs(save_path)

batch_size = 64

hidden_dim = 512
dropout = 0.5
lr = 1e-4
weight_decay = 1e-5
num_layers = 8
no_up_limit = 16


def metrics(truth, pred):
    acc = accuracy_score(truth, pred)
    f1 = f1_score(truth, pred)
    pre = precision_score(truth, pred)
    rec = recall_score(truth, pred)
    return acc, f1, pre, rec


def train_one_epoch():
    model.train()
    all_truth = []
    all_pred = []
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch)
        label = batch['news'].label[:batch['news'].batch_size]
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()

        all_truth += label.to('cpu').tolist()
        all_pred += out.argmax(dim=-1).to('cpu').tolist()


@torch.no_grad()
def validation(loader):
    model.eval()
    all_truth = []
    all_pred = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        label = batch['news'].label[:batch['news'].batch_size]

        all_truth += label.to('cpu').tolist()
        all_pred += out.argmax(dim=-1).to('cpu').tolist()
    return metrics(all_truth, all_pred)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = get_data()
    news_index = json.load(open('../news_index.json'))
    train_mask = json.load(open('{}_train.json'.format(dataset_name)))
    val_mask = json.load(open('{}_val.json'.format(dataset_name)))
    test_mask = json.load(open('{}_test.json'.format(dataset_name)))
    train_mask = torch.tensor([news_index[item] for item in train_mask], dtype=torch.long)
    val_mask = torch.tensor([news_index[item] for item in val_mask], dtype=torch.long)
    test_mask = torch.tensor([news_index[item] for item in test_mask], dtype=torch.long)
    train_loader = NeighborLoader(data,
                                  num_neighbors=[256] * 16,
                                  batch_size=64,
                                  input_nodes=('news', train_mask),
                                  shuffle=True)
    val_loader = NeighborLoader(data,
                                num_neighbors=[256] * 16,
                                batch_size=64,
                                input_nodes=('news', val_mask),
                                shuffle=True)
    test_loader = NeighborLoader(data,
                                 num_neighbors=[256] * 16,
                                 batch_size=64,
                                 input_nodes=('news', test_mask),
                                 shuffle=True)

    while True:
        model = TaHid(hidden_dim=hidden_dim,
                      num_layers=num_layers,
                      dropout=dropout,
                      metadata=data.metadata(),
                      mode=mode).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        best_acc = 0
        best_model = model.state_dict()
        no_up = 0
        pbar = tqdm(ncols=0)
        epoch = 0
        while True:
            train_one_epoch()
            val_acc, val_f1, val_pre, val_rec = validation(val_loader)
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model.state_dict()
                no_up = 0
            else:
                no_up += 1
            epoch += 1
            pbar.set_postfix_str('epoch {}, no up {}, val acc {}'.format(epoch, no_up, best_acc))
            if no_up == no_up_limit:
                break
        model.load_state_dict(best_model)
        acc, f1, pre, rec = validation(test_loader)
        torch.save(best_model, osp.join(save_path, '{:.6f}_{:.6f}_{:.6f}_{:.6f}.pt'.format(acc, f1, pre, rec)))



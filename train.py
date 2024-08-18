import json
import os
import torch
import torch.nn as nn
from dataset import obtain_hetero_data
from torch_geometric.loader import NeighborLoader
from model import FNDPro
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--ablation')
parser.add_argument('--dataset')
parser.add_argument('--mode', type=str, default='GAT')
args = parser.parse_args()
dataset_name = args.dataset
mode = args.mode
ablation = args.ablation
assert dataset_name in ['politifact', 'gossipcop', 'mixed']  # dataset
assert mode in ['RGCN', 'GCN', 'GAT', 'HGT']  # gnn type
assert ablation in ['last', 'mean', 'max', 'mlp', 'rnn', 'first', None]  # the fusion methods


# parameters for model and training
batch_size = 64
lr = 1e-4
weight_decay = 1e-5

hidden_dim = 512
dropout = 0.5
num_layers = 8


def metrics(truth, pred):
    acc = accuracy_score(truth, pred) * 100
    f1 = f1_score(truth, pred) * 100
    pre = precision_score(truth, pred) * 100
    rec = recall_score(truth, pred) * 100
    return acc, f1, pre, rec


def train_one_epoch(model, loss_fn, optimizer):
    model.train()
    all_truth = []
    all_pred = []
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch, ablation)
        label = batch['news'].label[:batch['news'].batch_size]
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()

        all_truth += label.to('cpu').tolist()
        all_pred += out.argmax(dim=-1).to('cpu').tolist()


@torch.no_grad()
def validation(model, loader):
    model.eval()
    all_truth = []
    all_pred = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch, ablation)
        label = batch['news'].label[:batch['news'].batch_size]

        all_truth += label.to('cpu').tolist()
        all_pred += out.argmax(dim=-1).to('cpu').tolist()
    return metrics(all_truth, all_pred)


def train(train_name):
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    save_path = f'checkpoints/{train_name}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model = FNDPro(hidden_dim=hidden_dim,
                   num_layers=num_layers,
                   dropout=dropout,
                   metadata=data.metadata(),
                   mode=mode,
                   ablation=ablation).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0
    best_state = model.state_dict()
    for key, value in best_state .items():
        best_state[key] = value.clone()
    no_up = 0
    no_up_limit = 8
    pbar = tqdm(ncols=0)
    epoch = 0
    while True:
        train_one_epoch(model, loss_fn, optimizer)
        val_acc, val_f1, val_pre, val_rec = validation(model, val_loader)
        if val_acc > best_acc:
            best_acc = val_acc
            for key, value in model.state_dict().items():
                best_state[key] = value.clone()
            no_up = 0
        else:
            no_up += 1
        epoch += 1
        pbar.set_postfix_str('epoch {}, no up {}, val acc {}'.format(epoch, no_up, best_acc))
        if no_up == no_up_limit:
            break
    model.load_state_dict(best_state)
    acc, f1, pre, rec = validation(model, test_loader)
    torch.save(best_state, '{}/{:.2f}_{:.2f}_{:.2f}_{:.2f}'.format(save_path, acc, f1, pre, rec))
    print(f'Best test accuracy: {acc:.2f}')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'data'
    data = obtain_hetero_data(data_dir=data_dir)  # here you could set which media you want to employ
    news_index = json.load(open(f'{data_dir}/news_index.json'))
    train_mask = json.load(open(f'split/{dataset_name}_train.json'))
    val_mask = json.load(open(f'split/{dataset_name}_train.json'))
    test_mask = json.load(open(f'split/{dataset_name}_train.json'))

    train_mask = torch.tensor([news_index[item] for item in train_mask], dtype=torch.long)
    val_mask = torch.tensor([news_index[item] for item in val_mask], dtype=torch.long)
    test_mask = torch.tensor([news_index[item] for item in test_mask], dtype=torch.long)

    # dataloader to process the huge news propagation network
    train_loader = NeighborLoader(data,
                                  num_neighbors=[256] * 16,
                                  batch_size=64,
                                  input_nodes=('news', train_mask),
                                  shuffle=True)
    val_loader = NeighborLoader(data,
                                num_neighbors=[256] * 16,
                                batch_size=64,
                                input_nodes=('news', val_mask),
                                shuffle=False)
    test_loader = NeighborLoader(data,
                                 num_neighbors=[256] * 16,
                                 batch_size=64,
                                 input_nodes=('news', test_mask),
                                 shuffle=False)
    name = f'{dataset_name}_{ablation}_{mode}'
    for i in range(10):
        train(name)





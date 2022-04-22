from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--num_samples', type=int, default=None,
                    help='Number of samples in a batch.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data: https://relational.fit.cvut.cz/dataset/CORA
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# adj: [2708, 2708]
# features: [2708, 1433]

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
#%%
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

adj_dense = adj.to_dense()
def train(epoch, num_samples = None):
    if num_samples is None:
        num_samples = features.size(0)
    t = time.time()
    model.train()
    s = features.size(0)/num_samples
    for _ in range(features.size(0)//num_samples):
        # get a mini-batch and select the training samples
        idx_sample = torch.randperm(features.size(0),device=features.device)[:num_samples]
        selected_train = (idx_sample.reshape(-1,1) == idx_sample).any(1)
        labels1 = labels[idx_sample]
        adj1 = adj_dense[idx_sample][:,idx_sample].to_sparse()
        optimizer.zero_grad()
        output = model(features[idx_sample], adj1,s=s)
        loss_train = F.nll_loss(output[selected_train], labels1[selected_train])*s
        acc_train = accuracy(output[selected_train], labels1[selected_train])
        loss_train.backward()
        optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    res = args.__dict__.copy()
    res.update({
        'loss':loss_test.item(),
        'acc': acc_test.item()
    })
    
    with open('results.txt','a+') as f:
        f.write(str(res))
        f.write('\n')



# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch,args.num_samples)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
# Epoch: 0200 loss_train: 0.4095 acc_train: 0.9286 loss_val: 0.6867 acc_val: 0.8167 time: 0.0093s
# Test set results: loss= 0.7265 accuracy= 0.8300
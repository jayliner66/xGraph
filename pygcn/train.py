from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN
from torch.autograd import Variable

import matplotlib, matplotlib.pyplot as plt
from matplotlib import cm
from colorspacious import cspace_converter

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
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

# Load data
adj, features, labels, idx_train, idx_val, idx_test, idx_all = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    idx_all = idx_all.cuda()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
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

# var_features = Variable(features, requires_grad = True)
# var_adj = Variable(adj, requires_grad = True)
    

# def posttest():
#     for i in range(1,2):
        # feature_mask = torch.zeros([features.shape[0], features.shape[1]], dtype = torch.float32)
        # feature_mask[i][:] += 1
        # adj_mask = torch.zeros([adj.shape[0], adj.shape[1]], dtype = torch.float32)
        # adj_mask[i][:] += 1
        # part_var_features = feature_mask * var_features + (1-feature_mask) * fixed_features
        # part_var_adj = adj_mask * var_adj + (1-adj_mask) * fixed_adj
        # model.train()
        # optimizer.zero_grad()
        # feature_row = Variable(features[i], requires_grad = True)
        # adj_row = Variable(adj[i], requires_grad = True)
        # output = model(features, adj, feature_row, adj_row, i)
        # highest_prob = torch.exp(output.max(1)[0])
        # given_prob = highest_prob[i]
        # t=time.time()
        # given_prob.backward(retain_graph = True)
        # print(time.time()-t)
        # print(feature_row.grad)
        # print(adj_row.grad)
    # highest_prob = torch.exp(output[idx_all].max(1)[0])
    # # loss_posttest = F.nll_loss(output[idx_all], labels[idx_all])
    # counter = 0
    # for i in range(labels.shape[0]):
    #     t = time.time()
    #     given_prob = highest_prob[i]
    #     print('time: {:.4f}s'.format(time.time() - t))
    #     given_prob.backward(retain_graph = True)
    #     print('time: {:.4f}s'.format(time.time() - t))
    #     grad1 = var_features.grad
    #     grad2 = var_adj.grad
    #     print('time: {:.4f}s'.format(time.time() - t))
    #     counter += 1
    #     print(counter)
    # print("DONE")

    # highestprob.backward()
    # print(var_features.grad)
    # print(var_adj.grad)
    # torch.set_printoptions(profile="full")
    # print(preds)
    # print(labels)
    # loss_posttest = F.nll_loss(output[idx_all], labels[idx_all])
    # loss_posttest.backward()
    # print(var_features.grad)
    # print(var_adj.grad)

def bary_to_cart(zerotharr, firstarr, secondarr):
    n = len(zerotharr)
    x = [firstarr[i]+.5*secondarr[i] for i in range(n)]
    y = [.8660254*secondarr[i] for i in range(n)]
    return x,y

def basicplot():
    output = model(features, adj)
    probabilities = torch.exp(output).detach().numpy()

    zerotharr0 = []
    firstarr0 = []
    secondarr0 = []
    zerotharr1 = []
    firstarr1 = []
    secondarr1 = []
    zerotharr2 = []
    firstarr2 = []
    secondarr2 = []
    
    for i in range(2000):
        if(labels[i] == 0):
            zerotharr0.append(probabilities[i][0])
            firstarr0.append(probabilities[i][1])
            secondarr0.append(probabilities[i][2])
        if(labels[i] == 1):
            zerotharr1.append(probabilities[i][0])
            firstarr1.append(probabilities[i][1])
            secondarr1.append(probabilities[i][2])
        if(labels[i] == 2):
            zerotharr2.append(probabilities[i][0])
            firstarr2.append(probabilities[i][1])
            secondarr2.append(probabilities[i][2])
    
    x0, y0 = bary_to_cart(zerotharr0, firstarr0, secondarr0)
    x1, y1 = bary_to_cart(zerotharr1, firstarr1, secondarr1)
    x2, y2 = bary_to_cart(zerotharr2, firstarr2, secondarr2)

    colors = ['r', 'g', 'b']

    type0 = plt.scatter(x0, y0, s=1, color=colors[0])
    type1 = plt.scatter(x1, y1, s=1, color=colors[1])
    type2 = plt.scatter(x2, y2, s=1, color=colors[2])

    plt.plot([0, 1], [0, 0], color='k', linestyle='-', linewidth=2)
    plt.plot([1, .5], [0, .8660254], color='k', linestyle='-', linewidth=2)
    plt.plot([.5, 0], [.8660254, 0], color='k', linestyle='-', linewidth=2)
    plt.plot([.5, .5], [0, .2886751], color='0.5', linestyle='-', linewidth=2)
    plt.plot([.75, .5], [.4330127, .2886751], color='0.5', linestyle='-', linewidth=2)
    plt.plot([.25, .5], [.4330127, .2886751], color='0.5', linestyle='-', linewidth=2)

    plt.legend((type0, type1, type2),
            ('Diabetes Mellitus, Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2'),
            scatterpoints=1,
            loc='upper right',
            ncol=1,
            fontsize=8)

    plt.show()

def posttest():
    var_features = Variable(features, requires_grad = True)
    var_adj = Variable(adj, requires_grad = True)
    model.train()
    optimizer.zero_grad()
    output = model(var_features, var_adj)
    highest_prob = torch.exp(output.max(1)[0])
    x = []
    for i in range(30):
        # if output.max(1)[1][i]==0:
        t = time.time()
        given_prob = highest_prob[i]
        given_prob.backward(retain_graph = True)
        fea_grad = var_features.grad[i]
        adj_grad = var_adj.grad[i]
        # for i in [0,8,184,15,286]:
        #     x.append(fea_grad[i].item())
        # for i in [89,251,70,346,444]:
        #     x.append(fea_grad[i].item())
        top_words = sorted(range(len(fea_grad)), key=lambda i: -fea_grad[i])[:10]
        top_articles = sorted(range(len(adj_grad)), key = lambda i: -adj_grad[i])[:5]
        bottom_words = sorted(range(len(fea_grad)), key=lambda i: fea_grad[i])[:10]
        bottom_articles = sorted(range(len(adj_grad)), key = lambda i: adj_grad[i])[:5]
        # Articles, features, labels are all zero indexed
        print('Article: {:05d}'.format(i),
        'Orig Label: {:01d}'.format(labels[i]),
        'Comp Label: {:01d}'.format(output.max(1)[1][i]),
        'Confidence: {:.4f}'.format(given_prob),
        'Top words:', top_words,
        'Bottom words:', bottom_words,
        # 'Top words:', [fea_grad[i] for i in top_words],
        # 'Bottom words', [fea_grad[i] for i in bottom_words],
        # 'Top articles:', [adj_grad[i] for i in top_articles],
        # 'Bottom articles', [adj_grad[i] for i in bottom_articles],
        'Time: {:.2f}s'.format(time.time() - t))
    # arr = np.array(x)
    # arr2 = np.reshape(arr, (-1,10))
    # print(arr2)

    # fig, ax = plt.subplots()
    # words = ["A","B","C","D","E","F","G","H","I","J"]
    # for i in range(arr2.shape[0]):
    #     x = [i for j in range(10)]
    #     y = arr2[i]
    #     ax.scatter(x, y, c=[j for j in range(10)], cmap = matplotlib.cm.get_cmap('tab10'))
    # ax.legend()
    # ax.grid(True)

    # plt.show()


def posttestchange():
    var_features = Variable(features, requires_grad = True)
    var_adj = Variable(adj, requires_grad = True)
    model.train()
    optimizer.zero_grad()
    output = model(var_features, var_adj)
    highest_prob = torch.exp(output.max(1)[0])
    for i in range(100):
        if output.max(1)[1][i] != labels[i]:
            t = time.time()
            given_prob = highest_prob[i]
            actual_prob = torch.exp(output[i][labels[i]])
            diff_prob = actual_prob - given_prob
            diff_prob.backward(retain_graph = True)
            fea_grad = var_features.grad[i]
            adj_grad = var_adj.grad[i]
            top_words = sorted(range(len(fea_grad)), key=lambda i: -fea_grad[i])[:10]
            # top_articles = sorted(range(len(adj_grad)), key = lambda i: -adj_grad[i])[:10]
            # Articles, features, labels are all zero indexed
            # if output.max(1)[1][i]==0:
            if True:
                print('Article: {:05d}'.format(i),
                'Orig Label: {:01d}'.format(labels[i]),
                'Comp Label: {:01d}'.format(output.max(1)[1][i]),
                'Given prob: {:.4f}'.format(given_prob),
                'Actual prob: {:.4f}'.format(actual_prob),
                'Top words:', top_words,
                # 'Top articles:', top_articles,
                'Time: {:.2f}s'.format(time.time() - t))

#hyperparameters for pertinent positives/negatives
kappa = 0
c = 1
beta = 0.001

def f_neg(i, x_0, delta_neg, kappa):
    full_delta = torch.zeros([features.shape[0], features.shape[1]], dtype = torch.float32)
    full_delta[i] += delta_neg
    output = model(x_0, adj)
    best_label = output.max(1)[1][i]
    output2 = model(x_0+full_delta, adj)
    new_highest_value = torch.exp(output2[i][best_label])

    j = (best_label+1)%3
    k = (best_label+2)%3
    # new_highest_value.backward(retain_graph = True)
    # print("HI", delta_neg.grad)

    # print(new_highest_value)
    # print(second_highest)

    second_highest = max(torch.exp(output2[i][j]), torch.exp(output2[i][k]))
    return max(new_highest_value-second_highest, -kappa)

    #NOW GET SECOND HIGHEST, COMPARE TO -KAPPA

def g_neg(i, x_0, delta_neg, kappa):
    return f_neg(i, x_0, delta_neg, kappa)+torch.norm(delta_neg)*torch.norm(delta_neg)

def S_beta(z, beta):
    return (z>beta)*(z-beta)+(z<-beta)*(z+beta)

def proj_neg(i, x_0, z):
    words = x_0[i]
    m = max(words)
    return (0<z)*(z<m)*(words==0)*z+(z>=m)*(words==0)*m*torch.ones(features.size()[1])

def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]

def fista_neg(i, x_0, delta_neg, y_neg, alpha, kappa, c, beta):
    steps = len(alpha)

    array_of_f_neg = []
    array_of_delta_neg = []
    array_of_penalty_neg = []
    t = time.time()

    for k in range(steps):
        y_neg = Variable(y_neg, requires_grad = True)
        prev_delta_neg = delta_neg
        # print("g: ", g_neg(i, x_0, y_neg, kappa))
        g_neg(i, x_0, y_neg, kappa).backward(retain_graph = True)
        # print("delta before: ", delta_neg)
        # print(y_neg.grad)
        # print(y_neg-alpha[k]*y_neg.grad)
        # print(S_beta(y_neg-alpha[k]*y_neg.grad, beta))
        delta_neg = proj_neg(i, x_0, S_beta(y_neg-alpha[k]*y_neg.grad, beta))
        # print("delta after: ", delta_neg)
        y_neg = proj_neg(i, x_0, delta_neg + (delta_neg - prev_delta_neg)*k/(k+3))
        array_of_f_neg.append(f_neg(i, x_0, delta_neg, kappa))
        array_of_delta_neg.append(delta_neg)
        array_of_penalty_neg.append((beta*torch.sum(delta_neg)+torch.norm(delta_neg)*torch.norm(delta_neg)))
        if(k%10==0):
            print(k, time.time()-t)
    # print(array_of_f_neg)
    # # print(delta_neg)
    # print(array_of_penalty_neg)
    # for i in range(steps):
    #     if array_of_f_neg[i] ==0:
    #         print(i)

    neg_total = [10000*array_of_f_neg[i]+array_of_penalty_neg[i] for i in range(steps)]
    # print(argmin(neg_total))
    pert_neg = array_of_delta_neg[argmin(neg_total)].detach()
    
    # return delta_neg, y_neg

    sorted_pert_neg = sorted(range(len(pert_neg)), key=lambda i: -pert_neg[i])[:10]
    print(sorted_pert_neg)


def f_pos(i, x_0, delta_pos, kappa):
    full_delta = torch.zeros([features.shape[0], features.shape[1]], dtype = torch.float32)
    full_delta[i] += delta_pos
    output = model(x_0, adj)
    best_label = output.max(1)[1][i]
    output2 = model(full_delta, adj)
    new_highest_value = torch.exp(output2[i][best_label])

    j = (best_label+1)%3
    k = (best_label+2)%3
    # new_highest_value.backward(retain_graph = True)
    # print("HI", delta_neg.grad)

    # print(new_highest_value)
    # print(second_highest)

    second_highest = max(torch.exp(output2[i][j]), torch.exp(output2[i][k]))
    return max(second_highest-new_highest_value, -kappa)

    #NOW GET SECOND HIGHEST, COMPARE TO -KAPPA

def g_pos(i, x_0, delta_pos, kappa):
    return f_pos(i, x_0, delta_pos, kappa)+torch.norm(delta_pos)*torch.norm(delta_pos)

def proj_pos(i, x_0, z):
    words = x_0[i]
    m = max(words)
    return (0<z)*(z<m)*(words>0)*z+(z>=m)*(words>0)*m*torch.ones(features.size()[1])

def fista_pos(i, x_0, delta_pos, y_pos, alpha, kappa, c, beta):
    steps = len(alpha)

    array_of_f_pos = []
    array_of_delta_pos = []
    array_of_penalty_pos = []
    t = time.time()

    for k in range(steps):
        y_pos = Variable(y_pos, requires_grad = True)
        prev_delta_pos = delta_pos
        # print("g: ", g_neg(i, x_0, y_neg, kappa))
        g_pos(i, x_0, y_pos, kappa).backward(retain_graph = True)
        # print("delta before: ", delta_neg)
        # print(y_neg.grad)
        # print(y_neg-alpha[k]*y_neg.grad)
        # print(S_beta(y_neg-alpha[k]*y_neg.grad, beta))
        delta_pos = proj_pos(i, x_0, S_beta(y_pos-alpha[k]*y_pos.grad, beta))
        # print("delta after: ", delta_neg)
        y_pos = proj_pos(i, x_0, delta_pos + (delta_pos - prev_delta_pos)*k/(k+3))
        array_of_f_pos.append(f_pos(i, x_0, delta_pos, kappa))
        array_of_delta_pos.append(delta_pos)
        array_of_penalty_pos.append((beta*torch.sum(delta_pos)+torch.norm(delta_pos)*torch.norm(delta_pos)))
        if (k%10==0):
            print(k, time.time()-t)
    # print(array_of_f_pos)
    # # print(delta_neg)
    # print(array_of_penalty_pos)
    # for i in range(steps):
    #     if array_of_f_pos[i] ==0:
    #         print(i)

    pos_total = [10000*array_of_f_pos[i]+array_of_penalty_pos[i] for i in range(steps)]
    # print(argmin(pos_total))
    pert_pos = array_of_delta_pos[argmin(pos_total)].detach()
    
    # return delta_neg, y_neg

    sorted_pert_pos = sorted(range(len(pert_pos)), key=lambda i: -pert_pos[i])[:10]
    print(sorted_pert_pos)




# def maskedtest(i):
#     masked_features = torch.randn(features.size()[0], features.size()[1])
#     masked_adj = torch.randn(adj.size()[0], adj.size()[1])
#     var_masked_features = Variable(masked_features, requires_grad = True)
#     var_masked_adj = Variable(masked_adj, requires_grad = True)
#     new_features = torch.mul(torch.sigmoid(var_masked_features), features)
#     new_adj = torch.mul(torch.sigmoid(var_masked_adj), adj)
#     model.train()
#     optimizer.zero_grad()
#     output = model(new_features, new_adj)
#     highest_prob = torch.exp(output.max(1)[0][i])
#     print(highest_prob)

#     highest_prob.backward()
#     print(features.grad)
#     print(adj.grad)
#     print(masked_features.grad)
#     print(masked_adj.grad)
    
# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# # # Testing
test()

# # # # # basicplot()

iterations = 200
step_sizes = []
for i in range(iterations):
    step_sizes.append(.02)
fista_neg(1, features, torch.zeros(features.size()[1]), torch.zeros(features.size()[1]), step_sizes, kappa, c, beta)
fista_pos(1, features, torch.zeros(features.size()[1]), torch.zeros(features.size()[1]), step_sizes, kappa, c, beta)

# # # print(features[0])
# posttest()

for i in [449, 444, 139, 484, 401, 379, 274, 477, 445, 451]:
    print(features[26][i])

# for i in [0,8,184,15,286,185,159,21,39,23]:
#     print('0: ', features[0][i])
# for i in [16, 346, 235, 359, 69, 418, 239, 212, 450, 253]:
#     print(features[3][i])
# # for i in [449, 444, 484, 139, 379, 401, 477, 274, 445, 451]:
# #     print('3: ', features[3][i])
# # for i in [16, 346, 235, 359, 418, 69, 239, 123, 212, 329]:
# #     print('6: ', features[6][i])

# # for i in [449, 444, 484, 139, 379, 401, 274, 445, 477, 451]:
# #     print('15: ', features[15][i])
# # for i in [0, 8, 15, 16, 21, 32, 38, 39, 24, 33]:
# #     print(features[0][i])
# # for i in [16,235,123,212,213,218,211,226]:
# #     print(features[6][i])
# # print(features[0])
# # print(features[1])
# # print(features[2])
# print(features[3])
# print(features[15])
# # print(features[4])

# posttest()

# maskedtest(0)

#Given graph adjacency matrix, use page rank

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
import networkx as nx
import ndf
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class GraphRfi(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e, features_num):
        super(GraphRfi, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim
        self.features_num = features_num
        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss(reduce=False)

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u)
        #print(embeds_u.shape)
        embeds_v = self.enc_v_history(nodes_v)
        self.x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        self.x_u = F.dropout(self.x_u, training=self.training)
        self.x_u = self.w_ur2(self.x_u)  # the embeddings of nodes_u
        self.x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        self.x_v = F.dropout(self.x_v, training=self.training)
        self.x_v = self.w_vr2(self.x_v)
        x_uv = torch.cat((self.x_u, self.x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        #print(x)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        #loss_fn = nn.MSELoss(reduce=False)
        #lo = loss_fn(scores, labels_list)
        #print(lo.mean())
        #print(self.criterion(scores, labels_list))
        return self.criterion(scores, labels_list)

    def get_embedding(self, nodes_u):
        embeds_u = self.enc_u(nodes_u)
        return embeds_u
    def get_v_embedding(self, nodes_v):
        embeds_v = self.enc_v_history(nodes_v)
        return embeds_v

def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae, test_loss, correct, correctness
          , labels, neuralforest, args, history_u_lists, history_ur_lists):
    neuralforest.train()
    model.train()
    running_loss = 0.0 
    corr = 0
    criterion = nn.MSELoss(reduce=False)
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data 
        nodes = batch_nodes_u
        embed_u = model.get_embedding(nodes).cuda()
        errors = []
        for k in nodes:
            items = torch.Tensor(history_u_lists[k.item()]).long()
            ratings = torch.Tensor(history_ur_lists[k.item()]).float()
            if len(items) == 1:
                items = torch.cat((items, items), dim=0)
                ratings = torch.cat((ratings, ratings), dim=0)
            ones = (torch.Tensor(np.ones(len(items))) * k).long()
            error = model.loss(ones.to(device), items.to(device), ratings.to(device)).mean().item()/10
            errors.append(error)
        errors = torch.Tensor(np.array(errors)).float()
        errors = errors.view(len(batch_nodes_u), 1).cuda()
        embed_u = torch.cat((embed_u, errors), dim=1).cuda()
        label_tensor = labels[nodes].long().cuda()
        output = neuralforest(embed_u)
        pro = output[:, 1]
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        corr += pred.eq(label_tensor.data.view_as(pred)).cuda().sum()
        optimizer.zero_grad()
        loss1 = neuralforest.loss(output.to(device), label_tensor.to(device))
        loss2 = (model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))*pro).mean()
        loss = 5 * loss1 + loss2
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, corr, len(train_loader.dataset), loss1.item()))
    return 0

def test(model, device, test_loader, labels, neuralforest):
    model.eval()
    neuralforest.eval()
    tmp_pred = []
    target = []
    test_loss = 0
    correct = 0
    ll = 0
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            nodes = test_u
            ll += len(nodes)
            embed_u = model.get_embedding(nodes)#
            label = labels[nodes].long()#
            embed_u = Variable(embed_u)
            label = Variable(label)
            #output = neuralforest(embed_u)
            #test_loss += F.nll_loss(torch.log(output), label, size_average=False).item()  # sum up batch loss
            #pred = output.data.max(1, keepdim=True)[1]
            #correct += pred.eq(label.data.view_as(pred)).cpu().sum()
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    test_loss /= len(test_loader.dataset)
    return expected_rmse, mae, test_loss, correct, ll


def split_data(data, G, G_f, fake_edges):
    feature_map, features_num, label_map = read_feat()
    train_set = set()
    test_set = set()
    user_set = set()
    item_set = set()
    item_train = set()
    empty = set()
    fake_users = set()
    fake_de = set()
    for node in G.nodes():
        adj = list(G.adj[node])
        if G.nodes[node]['class'] == "user" and len(adj) >= 5:
            user_set.add(node)
            train_count = len(adj) * 0.8
            train_count = int(train_count)
            for i in adj[0:train_count]:
                item_set.add(i)
                item_train.add(i)
                train_set.add((node, i))
            for j in adj[train_count:]:
                test_set.add((node, j))
                item_set.add(j)
    for edge in test_set:
        user, item = edge
        if item not in item_train:
            train_set.add((user, item))
            empty.add((user, item))
    for edge in empty:
        test_set.remove(edge)
    print("real users: ", len(user_set))
    for edge in fake_edges:
        user, item = edge
        fake_users.add(user)
        fake_de.add((user,item))
        train_set.add((user, item))
        user_set.add(user)
        item_set.add(item)
        rating = G_f[user][item]['weight']
        label = G_f.nodes[user]["label"]
        G.add_edge(user, item, weight=rating)
        G.nodes[user]['label'] = label
        G.nodes[user]['class'] = "user"
        G.nodes[item]['class'] = "item"
    print("train:", len(train_set), "test: ",len(test_set), "users: ", len(user_set),
          "items: ", len(item_set), "fake users: ", len(fake_users), "fake items: ", len(fake_de))
    return train_set, test_set, user_set, item_set, G
    #return sorted(train_set), sorted(test_set), user_set, item_set, G
def read_feat():
    filename = "data/feature_yelp.npy"
    features = np.load(filename)
    feature_map = {}
    features_num = len(features[0][2:])
    label_map = {}
    for i in features:
        user = i[0]
        label = i[1]
        label_map[user] = label
        feature = i[2:]
        feature_map[user] = feature
        #feature_map[user]["label"] = label
    return feature_map, features_num, label_map


def read_file(filename):
    with open(filename, 'r', encoding="utf8") as f:
        lines = f.readlines()
    f.close()
    G = nx.Graph()
    G_train = nx.Graph()
    G_f = nx.DiGraph()
    edges = set()
    fake_edges = set()
    user_fake_sets = set()
    for line in lines:
        line = line.strip().split("\t")
        user = line[0]
        item = line[1]
        rating = float(line[2])-1
        rating = int(rating)
        label = line[3]
        #print(label)
        if label == '1':
            G.add_edge(user, item, weight=rating)
            G.nodes[user]['label'] = label
            G.nodes[user]['class'] = "user"
            G.nodes[item]['class'] = "item"
            edges.add((user, item))
        elif label == '-1':
            G_f.add_edge(user, item, weight=rating)
            G_f.nodes[user]['label'] = label
            G_f.nodes[user]['class'] = "user"
            G_f.nodes[item]['class'] = "item"

            #fake_edges.add((user, item))
    ############reindex##############
    user_fake = set()
    for user, item in G_f.edges():
        if G_f.degree[user] > 5:
            user_fake.add(user)
            rating = G_f[user][item]['weight']
            label = G_f.nodes[user]["label"]
            #G.add_edge(user, item, weight=rating)
            #G.nodes[user]['label'] = label
            #G.nodes[user]['class'] = "user"
            #G.nodes[item]['class'] = "item"
            fake_edges.add((user, item))
    user_map = {}
    item_map = {}
    u_count = 0
    v_count = 0
    history_u_lists = defaultdict(list)
    history_ur_lists = defaultdict(list)
    history_v_lists = defaultdict(list)
    history_vr_lists = defaultdict(list)
    social_adj_lists = defaultdict(set)
    train_u = []
    train_v = []
    train_r = []
    test_u = []
    test_v = []
    test_r = []
    ratings_list = [0, 1, 2, 3, 4]
    f_len = len(fake_edges)
    print("Fake edges:", f_len)
    train_data, test_data, user_sets, item_sets, G = split_data(edges, G, G_f, list(sorted(fake_edges))[0:int(0.2*f_len)])
    adj_matrix = np.zeros((len(user_sets), len(item_sets)))

    for node in user_sets:
        user_map[node] = u_count
        u_count += 1

    for item in item_sets:
        item_map[item] = v_count
        v_count += 1

    for edge in train_data:
        user, item = edge
        train_u.append(user_map[user])
        train_v.append(item_map[item])
        rating = G[user][item]['weight']
        adj_matrix[user_map[user], item_map[item]] = rating
        train_r.append(rating)
        G_train.add_edge(user, item, weight=rating)
        G_train.nodes[user]['label'] = label
        G_train.nodes[user]['class'] = "user"
        G_train.nodes[item]['class'] = "item"
    #print("nonezero:", len(np.nonzero(adj_matrix)[0]))
    for edge in test_data:
        user, item = edge
        test_u.append(user_map[user])
        test_v.append(item_map[item])
        rating = G[user][item]['weight']
        test_r.append(rating)
    for node in G_train.nodes():
        if G_train.nodes[node]['class'] == "user":
            di = []
            re = []
            for item in G_train[node]:
                rating = G_train[node][item]['weight']
                re.append(rating)
                di.append(item_map[item])
            history_u_lists[user_map[node]] = di
            history_ur_lists[user_map[node]] = re
        if G_train.nodes[node]['class'] == "item":
            di = []
            re = []
            for item in G_train[node]:
                rating = G_train[item][node]['weight']
                re.append(rating)
                di.append(user_map[item])
            history_v_lists[item_map[node]] = di
            history_vr_lists[item_map[node]] = re
    feature_map, features_num, label_map = read_feat()
    label_new = np.zeros(len(user_sets))
    feature_new = np.zeros((len(user_sets), features_num))
    xxxxx = {'-1':0,'1':1}
    for i in user_sets:
        label_new[user_map[i]] = xxxxx[label_map[i]]
        feature_new[user_map[i]] = feature_map[i]
        #feature_new[user_map[i]]["label"]  = feature_map[i]["label"]
    #feature_new = preprocessing.scale(feature_new)
    return history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, \
           test_v, test_r, ratings_list, feature_new, features_num, label_new, adj_matrix


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GraphRfi model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=100, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('-n_tree', type=int, default=80)
    parser.add_argument('-tree_depth', type=int, default=10)
    parser.add_argument('-n_class', type=int, default=2)
    parser.add_argument('-tree_feature_rate', type=float, default=0.5)
    parser.add_argument('-jointly_training', action='store_true', default=True)
    parser.add_argument('-feat_dropout', type=float, default=0.3)
    args = parser.parse_args()
    args.cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim
    dir_data = './data/yelp_final.txt'
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v\
        , test_r, ratings_list, feature_new, features_num, label_new, adj_matrix = read_file(dir_data)
    label_tensor = Variable(torch.Tensor(label_new))
    adj_matrix = Variable(torch.Tensor(adj_matrix))
    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()
    u2e = nn.Embedding(num_users, features_num).to(device)
    u2e.weight = nn.Parameter(torch.FloatTensor(feature_new), requires_grad=False)
    u2e.to(device)
    v2e = nn.Embedding(num_items, features_num).to(device)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)
    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, features_num, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, embed_dim, features_num, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)
    # item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, features_num, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e,  embed_dim, features_num, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)
    graphrfi = GraphRfi(enc_u_history, enc_v_history, r2e, features_num).to(device)

    feat_layer = ndf.UCIAdultFeatureLayer(args.feat_dropout)
    forest = ndf.Forest(n_tree=args.n_tree, tree_depth=args.tree_depth, n_in_feature=feat_layer.get_out_feature_size(),
                        tree_feature_rate=args.tree_feature_rate, n_class=args.n_class,
                        jointly_training=args.jointly_training)
    neuralforest = ndf.NeuralDecisionForest(feat_layer, forest).to(device)
    optimizer = torch.optim.Adam(list(graphrfi.parameters())+list(neuralforest.parameters()), lr=args.lr)
    #optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-5)
    #optimizer2 = torch.optim.RMSprop(graphrfi.parameters(), lr=args.lr, alpha=0.9)
    best_rmse = 9999.0
    best_mae = 9999.0
    test_loss = 9999.0
    correct = 9999.0
    correctness = 9999.0
    endure_count = 0

    for epoch in range(1, args.epochs + 1):
        train(graphrfi, device, train_loader, optimizer, epoch, best_rmse, best_mae, test_loss, correct, correctness
              , label_tensor, neuralforest, args, history_u_lists,history_ur_lists)
        expected_rmse, mae, test_loss, correct, correctness = test(graphrfi, device, test_loader, label_tensor, neuralforest)
        # please add the validation set to tune the hyper-parameters based on your datasets.
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
        else:
            endure_count += 1
        print("rmse: %.4f, mae:%.4f, loss:%.4f, correct:%.4f, correctness:%.4f " % (expected_rmse, mae, test_loss, correct, correctness))

        if endure_count > 5:
            break


if __name__ == "__main__":
    main()

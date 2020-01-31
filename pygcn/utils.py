import numpy as np
import scipy.sparse as sp
import torch
import csv


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def parse_node():
    node_reader = list(csv.DictReader(open('../data/diabetes_node.tab', newline=''), delimiter='\t'))
    header_dict = node_reader[0]
    words = [header_dict['paper']]+header_dict[None][:-1]
    num_rows = len(node_reader)-1
    num_columns = len(words)
    words_to_column = {}
    for i in range(len(words)):
        words_to_column[words[i][10:-4]]=i
    idx = np.zeros(num_rows, dtype = np.int32)
    labels = np.zeros(num_rows, dtype = np.int32)
    features = np.zeros((num_rows, num_columns), dtype=np.float32)
    for i in range(1, len(node_reader)):
        idx[i-1] = node_reader[i]['NODE']
        labels[i-1] = node_reader[i]['paper'][-1]
        feature_row = node_reader[i][None][:-1]
        for word_weight in feature_row:
            [word, weight] = word_weight.split('=')
            word_num = words_to_column[word[2:]]
            weight = float(weight)
            features[i-1][word_num] = weight
    print('Diabetes Mellitus, Experimental: ', [words[i][10:-4] for i in [0,8,184,15,286,285,159,21,39,23, 93]])
    print('Diabetes Mellitus Type 1: ', [words[i][10:-4] for i in [449, 444, 484, 139, 379, 401, 477, 274, 445, 451]])
    print('Diabetes Mellitus Type 2: ', [words[i][10:-4] for i in [16, 346, 235, 359, 418, 69, 239, 123, 212, 329]])
    print([words[i][10:-4] for i in [0, 8, 15, 16, 21, 32, 38, 39, 24, 33]])
    print([words[i][10:-4] for i in [70, 89, 139, 247, 251, 379, 401, 444, 449, 484]])
    print("none")
    print([words[i][10:-4] for i in [16, 346, 235, 359, 69, 418, 239, 212, 450, 253]])
    print([words[i][10:-4] for i in [16, 235, 123, 212, 213, 218, 211, 226]])
    print([words[i][10:-4] for i in [449, 444, 484, 401, 379, 477, 274, 445, 482, 451]])
    
    return idx, labels, features

def parse_directed():
    directed_reader = list(csv.DictReader(open('../data/diabetes_directed.tab', newline=''), delimiter='\t'))
    num_rows = len(directed_reader)-1
    edges_unordered = np.zeros((num_rows, 2), dtype = np.int32)
    for i in range(1, len(directed_reader)):
        edge_1 = directed_reader[i]['cites'][6:]
        edge_2 = directed_reader[i][None][1][6:]
        edges_unordered[i-1][0] = int(edge_1)
        edges_unordered[i-1][1] = int(edge_2)
    return edges_unordered

def load_data():
    """Load citation network dataset (cora only for now)"""
    print('Loading PubMed diabetes dataset...')

    pre_idx, pre_labels, pre_features = parse_node()
    pre_edges_unordered = parse_directed()
    features = sp.csr_matrix(pre_features)
    labels = encode_onehot(pre_labels)

    # build graph
    idx = pre_idx
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = pre_edges_unordered
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    # adj = np.zeros((labels.shape[0], labels.shape[0]]), dtype = np.float32)
    # for i in edges.shape[0]:
    #     adj[edges[i][0]][edges[i][1]] += 1
    #     adj[edges[i][1]][edges[i][0]] += 1
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    # for i in range(labels.shape[0]):
    #     adj[i][i] += 1
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(8000)
    idx_val = range(8000, 15000)
    idx_test = range(15000, 18000)
    idx_all = range(labels.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = adj.to_dense()

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    idx_all = torch.LongTensor(idx_all)

    return adj, features, labels, idx_train, idx_val, idx_test, idx_all

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

parse_node()

# def parse_directed_as_networkx():
#     directed_reader = list(csv.DictReader(open('../data/diabetes_directed.tab', newline=''), delimiter='\t'))
#     num_rows = len(directed_reader)-1
#     edges_unordered = np.zeros((num_rows, 2), dtype = np.int32)
#     for i in range(1, len(directed_reader)):
#         edge_1 = directed_reader[i]['cites'][6:]
#         edge_2 = directed_reader[i][None][1][6:]
#         edges_unordered[i-1][0] = int(edge_1)
#         edges_unordered[i-1][1] = int(edge_2)
#     return edges_unordered

# def pagerank(G, alpha=0.85, personalization=None, 
#              max_iter=100, tol=1.0e-6, nstart=None, weight='weight', 
#              dangling=None): 
#     """Return the PageRank of the nodes in the graph. 
  
#     PageRank computes a ranking of the nodes in the graph G based on 
#     the structure of the incoming links. It was originally designed as 
#     an algorithm to rank web pages. 
  
#     Parameters 
#     ---------- 
#     G : graph 
#       A NetworkX graph.  Undirected graphs will be converted to a directed 
#       graph with two directed edges for each undirected edge. 
  
#     alpha : float, optional 
#       Damping parameter for PageRank, default=0.85. 
  
#     personalization: dict, optional 
#       The "personalization vector" consisting of a dictionary with a 
#       key for every graph node and nonzero personalization value for each node. 
#       By default, a uniform distribution is used. 
  
#     max_iter : integer, optional 
#       Maximum number of iterations in power method eigenvalue solver. 
  
#     tol : float, optional 
#       Error tolerance used to check convergence in power method solver. 
  
#     nstart : dictionary, optional 
#       Starting value of PageRank iteration for each node. 
  
#     weight : key, optional 
#       Edge data key to use as weight.  If None weights are set to 1. 
  
#     dangling: dict, optional 
#       The outedges to be assigned to any "dangling" nodes, i.e., nodes without 
#       any outedges. The dict key is the node the outedge points to and the dict 
#       value is the weight of that outedge. By default, dangling nodes are given 
#       outedges according to the personalization vector (uniform if not 
#       specified). This must be selected to result in an irreducible transition 
#       matrix (see notes under google_matrix). It may be common to have the 
#       dangling dict to be the same as the personalization dict. 
  
#     Returns 
#     ------- 
#     pagerank : dictionary 
#        Dictionary of nodes with PageRank as value 
  
#     Notes 
#     ----- 
#     The eigenvector calculation is done by the power iteration method 
#     and has no guarantee of convergence.  The iteration will stop 
#     after max_iter iterations or an error tolerance of 
#     number_of_nodes(G)*tol has been reached. 
  
#     The PageRank algorithm was designed for directed graphs but this 
#     algorithm does not check if the input graph is directed and will 
#     execute on undirected graphs by converting each edge in the 
#     directed graph to two edges. 
  
      
#     """
#     if len(G) == 0: 
#         return {} 
  
#     if not G.is_directed(): 
#         D = G.to_directed() 
#     else: 
#         D = G 
  
#     # Create a copy in (right) stochastic form 
#     W = nx.stochastic_graph(D, weight=weight) 
#     N = W.number_of_nodes() 
  
#     # Choose fixed starting vector if not given 
#     if nstart is None: 
#         x = dict.fromkeys(W, 1.0 / N) 
#     else: 
#         # Normalized nstart vector 
#         s = float(sum(nstart.values())) 
#         x = dict((k, v / s) for k, v in nstart.items()) 
  
#     if personalization is None: 
  
#         # Assign uniform personalization vector if not given 
#         p = dict.fromkeys(W, 1.0 / N) 
#     else: 
#         missing = set(G) - set(personalization) 
#         if missing: 
#             raise NetworkXError('Personalization dictionary '
#                                 'must have a value for every node. '
#                                 'Missing nodes %s' % missing) 
#         s = float(sum(personalization.values())) 
#         p = dict((k, v / s) for k, v in personalization.items()) 
  
#     if dangling is None: 
  
#         # Use personalization vector if dangling vector not specified 
#         dangling_weights = p 
#     else: 
#         missing = set(G) - set(dangling) 
#         if missing: 
#             raise NetworkXError('Dangling node dictionary '
#                                 'must have a value for every node. '
#                                 'Missing nodes %s' % missing) 
#         s = float(sum(dangling.values())) 
#         dangling_weights = dict((k, v/s) for k, v in dangling.items()) 
#     dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0] 
  
#     # power iteration: make up to max_iter iterations 
#     for _ in range(max_iter): 
#         xlast = x 
#         x = dict.fromkeys(xlast.keys(), 0) 
#         danglesum = alpha * sum(xlast[n] for n in dangling_nodes) 
#         for n in x: 
  
#             # this matrix multiply looks odd because it is 
#             # doing a left multiply x^T=xlast^T*W 
#             for nbr in W[n]: 
#                 x[nbr] += alpha * xlast[n] * W[n][nbr][weight] 
#             x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n] 
  
#         # check convergence, l1 norm 
#         err = sum([abs(x[n] - xlast[n]) for n in x]) 
#         if err < N*tol: 
#             return x 
#     raise NetworkXError('pagerank: power iteration failed to converge '
#                         'in %d iterations.' % max_iter) 
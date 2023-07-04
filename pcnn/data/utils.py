import torch
from torch_geometric.data import Batch
import scipy.sparse
import numpy as np
from scipy import sparse
from pcnn.data.scattering_utils import compute_scattering_features
import torch_geometric.transforms as T
from torch_geometric.transforms.knn_graph import KNNGraph
import torch_geometric

def laplacian_collate_fn(batch, follow_batch = None, exclude_keys = None):
    b = Batch.from_data_list(batch, follow_batch,
                                        exclude_keys)
    
    if hasattr(batch[0],"eigvec"):
        laplacians_eigvec = [data.eigvec for data in batch]
        L_coo = scipy.sparse.block_diag(laplacians_eigvec)

        values = L_coo.data
        indices = np.vstack((L_coo.row, L_coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = L_coo.shape

        L_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape)) 

        b.L = L_tensor
        del b.eigvec 
    
    return b


def compute_dist(X):
    # computes all (squared) pairwise Euclidean distances between each data point in X
    # D_ij = <x_i - x_j, x_i - x_j>
    G = np.matmul(X, X.T)
    D = np.reshape(np.diag(G), (1, -1)) + np.reshape(np.diag(G), (-1, 1)) - 2 * G
    return D

def compute_kernel(D, eps, d):
    # computes kernel for approximating GL
    # D is matrix of pairwise distances
    K = np.exp(-D/eps) * np.power(eps, -d/2)
    return K

def laplacian_epsilon_transform(eps, K, d = 2, eps_quantile=0.5, **kwargs):
    return lambda data : laplacian_epsilon(data, eps, K, d, eps_quantile)


def build_edge_idx(num_nodes):
    # Initialize edge index matrix
    E = torch.zeros((2, num_nodes * (num_nodes - 1)), dtype=torch.long)
    
    # Populate 1st row
    for node in range(num_nodes):
        for neighbor in range(num_nodes - 1):
            E[0, node * (num_nodes - 1) + neighbor] = node

    # Populate 2nd row
    neighbors = []
    for node in range(num_nodes):
        neighbors.append(list(np.arange(node)) + list(np.arange(node+1, num_nodes)))
    E[1, :] = torch.Tensor([item for sublist in neighbors for item in sublist])
    
    return E

def create_epsilon_graph(data, eps, d = 2, eps_quantile = 0.5, **kwargs):
    X = data.pos.numpy()
    n = X.shape[0]
    dists = compute_dist(X)
    if eps == "auto":
        triu_dists = np.triu(dists)
        eps = np.quantile(triu_dists[np.nonzero(triu_dists)], eps_quantile)
    W = compute_kernel(dists, eps, d)

    edge_index = build_edge_idx(n)
    edge_attr = W[edge_index[0],edge_index[1]]
    data.edge_index = edge_index
    data.edge_attr = torch.Tensor(edge_attr)
    return data

def epsilon_graph_transform(**kwargs):
    return lambda x: create_epsilon_graph(x, **kwargs)


def laplacian_epsilon(data, eps, K, d = 2, eps_quantile=0.5):
    # X is n x d matrix of data points
    X = data.pos.numpy()
    n = X.shape[0]
    dists = compute_dist(X)
    if eps == "auto":
        triu_dists = np.triu(dists)
        eps = np.quantile(triu_dists[np.nonzero(triu_dists)], eps_quantile)
    W = compute_kernel(dists, eps, d)
    D = np.diag(np.sum(W, axis=1, keepdims=False))
    L = sparse.csr_matrix(D - W)
    S, U = sparse.linalg.eigsh(L, k = K, which='SM')
    S = np.reshape(S.real, (1, -1))/(eps * n)
    S[0,0] = 0 # manually enforce this
    # normalize eigenvectors in usual l2 norm
    U = np.divide(U.real, np.linalg.norm(U.real, axis=0, keepdims=True))

    data.node_attr_eig = torch.from_numpy(S[0])
    data.eigvec = torch.from_numpy(U)
    data.eps = eps

    return data
    #return S, U, eps

def scattering_features_transform_(data,norm_list, J):
    features = compute_scattering_features(data,norm_list,J)
    data.scattering_features = torch.from_numpy(features)[None,...]
    return data

def scattering_features_transform(norm_list, J, **kwargs):
    return lambda data : scattering_features_transform_(data,norm_list,J)


def lap_transform(data):
    """
    Computing the laplacian from a graph and storing the eigenvalues and eigenvectors
    """
    L_sparse = torch_geometric.utils.get_laplacian(data.edge_index)
    L = torch_geometric.utils.to_dense_adj(L_sparse[0], edge_attr=L_sparse[1])
    eig, eigvec =  np.linalg.eigh(L)
    data.node_attr_eig = torch.from_numpy(eig[0])
    data.eigvec = torch.from_numpy(eigvec[0])
    return data


def get_pretransforms(compute_laplacian, graph_type, compute_scattering_feats, pre_transforms_base = None, **kwargs):

    if pre_transforms_base is None:
        pre_transforms = []
    else:
        pre_transforms = pre_transforms_base
        # T.NormalizeScale(), T.SamplePoints(display_sample)
    # scattering
    if graph_type == "knn":
        pre_transforms = pre_transforms + [ KNNGraph(kwargs["k"]) ]
    elif graph_type == "epsilon":
        pre_transforms = pre_transforms + [ epsilon_graph_transform(**kwargs) ] 
    
    if compute_laplacian == "epsilon":
        pre_transforms = pre_transforms + [ laplacian_epsilon_transform(**kwargs)]
    elif compute_laplacian == "combinatorial":
        pre_transforms = pre_transforms + [ lap_transform ]
     
    if compute_scattering_feats:
        pre_transforms = pre_transforms + [ scattering_features_transform(**kwargs)]

    return pre_transforms
    #GCN
    #graph_type = "knn"
    #compute_laplacian = False
    #compute_scattering_feats = False
    #pre_transforms = [ KNNGraph(kwargs["k"])  ]

    #MNN
    #graph_type = "knn"
    #compute_laplacian = "combinatorial"
    #compute_scattering_feats = False
    #pre_transforms = [ KNNGraph(kwargs["k"]), lap_transform ]

    #Scattering
    #graph_type = null
    #compute_laplacian = "epsilon"
    #compute_scattering_feats = True


import torch
import torch_geometric


def compute_sparse_diffusion_operator(b):
    #L_sparse = torch_geometric.utils.get_laplacian(b.edge_index)
    A_sparse = torch_geometric.utils.to_torch_coo_tensor(b.edge_index, b.edge_weight)
    D = A_sparse.sum(1).to_dense()
    Dinv = torch.sparse.spdiags(1/D.squeeze(), offsets = torch.zeros(1).long(),shape = (len(D),len(D)))
    P_sparse = torch.sparse.mm(Dinv,A_sparse)
    return P_sparse

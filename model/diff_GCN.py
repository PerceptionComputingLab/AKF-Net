import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, dropout=0.5):
        super(GCNConv, self).__init__()
        self.out_dropout = nn.Dropout(dropout)
        self.W = nn.Parameter(torch.rand(in_channels, out_channels, requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.rand(out_channels, requires_grad=True))
        else:
            self.bias = None
    
    def forward(self, X, adj):
        out = torch.matmul(torch.matmul(adj, X), self.W)
        if self.bias is not None:
            out += self.bias
        out = torch.relu(out)
        out = self.out_dropout(out)
        return out


# class GCNConv(nn.Module):
#     def __init__(self, in_channels, out_channels, bias=True, dropout=0.5):
#         super(GCNConv, self).__init__()
#         self.out_dropout = nn.Dropout(dropout)
#         self.W = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
#         if bias:
#             self.bias = nn.Parameter(torch.FloatTensor(out_channels))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.W.size(1))
#         self.W.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
    
#     def forward(self, X, adj):
#         out = torch.matmul(torch.matmul(adj, X), self.W)
#         if self.bias is not None:
#             out += self.bias
#         out = torch.relu(out)
#         out = self.out_dropout(out)
#         return out


class DifferentialGCNBlock(nn.Module):
    def __init__(self, frame_num, feature_shape, channel_dim, hidden_dim, dropout, topk=10):
        super().__init__()
        self.topk = topk
        self. feature_shape = feature_shape
        self.feature_num = feature_shape[0] * feature_shape[1] * feature_shape[2]
        # Compute the adjacency matrix of intra_GCN
        self.adj_space = self.get_adj_from_3Dspace(feature_shape)
        self.adj_frame = self.get_adj_frome_frame(frame_num=frame_num)

        self.intra_GCN_layer1 = GCNConv(in_channels=channel_dim, out_channels=channel_dim, bias=False, dropout=dropout)

        self.inter_GCN_layer1 = GCNConv(in_channels=channel_dim, out_channels=channel_dim, bias=False, dropout=dropout)

    
    def get_adj_from_3Dspace(self, feature_shape):
        '''
            input: 
            output: (feature_num, feature_num)
            The value of the diagonal is 1
        '''
        def __pos(h, w, d):
            h = np.clip(h, 0, H - 1)
            w = np.clip(w, 0, W - 1)
            d = np.clip(d, 0, D - 1)
            return h * (W * D) + w * (D) + d
        H, W, D = feature_shape
        adj_len = self.feature_num
        adj = torch.zeros((adj_len, adj_len), dtype=torch.int16)
        dirs = [-1, 0, 1]
        for i in range(H):
            for j in range(W):
                for k in range(D):
                    coordinate = __pos(i, j, k)
                    for dir_i in dirs:
                        for dir_j in dirs:
                            for dir_k in dirs:
                                adj[coordinate, __pos(i + dir_i, j + dir_j , k + dir_k)] = 1
        return adj

    def get_adj_from_cossim(self, feature):
        '''
            input: (bs, frame, feature_num, channel)
            output: (bs, frame, feature_num, feature_num)
            The value of the diagonal is 1
        '''
        similarity_score = torch.cosine_similarity(feature.unsqueeze(3).detach(), feature.unsqueeze(2).detach(), dim=-1).cpu().numpy() # (bs, frame, feature_num, feature_num)
        edges = []
        for i in range(self.feature_num):
            ind = np.argpartition(similarity_score[:, :, i, :], -(self.topk + 1))[:, :, -(self.topk + 1):]  
            node_i = np.full(ind.shape, i, dtype=ind.dtype)
            edge = np.concatenate([node_i[:, :, :, np.newaxis], ind[:, :, :, np.newaxis]], axis=-1)
            edges.append(edge)
        edges = np.concatenate(edges, axis=-2)  # (bs, frame, feature_num x (topk+1), 2)

        adj = torch.zeros(list(edges.shape[:2]) + [self.feature_num, self.feature_num]) # (bs, frame, feature_num, feature_num)
        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                for k in range(edges.shape[2]):
                    x, y = edges[i, j, k]
                    adj[i, j, x, y] = 1
        adj = adj + adj.permute(0, 1, 3, 2).multiply(adj.permute(0, 1, 3, 2) > adj) - adj.multiply(adj.permute(0, 1, 3, 2) > adj)
        return adj

    def get_adj_frome_frame(self, frame_num):
        '''
            input: 
            output: (frame_num, frame_num)
            The value of the diagonal is 0
        '''
        adj = torch.zeros((frame_num, frame_num), dtype=torch.int16)
        for i in range(frame_num-1):
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1
        return adj

    def normalize_adj(self, adj, add_eye=True):
        if add_eye:
            adj_hat = (adj + torch.eye(adj.size(-1)).to(adj.device)).to(dtype=torch.float32)
        else:
            adj_hat = adj.to(dtype=torch.float32)
        D_matrix = torch.sum(adj_hat, -1)
        D_matrix = torch.diag_embed(D_matrix).to(dtype=torch.float32)
        D_inv    = D_matrix.inverse().sqrt()
        adj_hat  = torch.matmul(torch.matmul(D_inv, adj_hat), D_inv)
        return adj_hat

    def forward(self, d_seq):
        '''
            input: (bs, frame, channel, H, W, D)
            output: (bs, frame, channel ,H, W, D)
        '''
        input_shape = d_seq.shape
        d_seq = torch.reshape(d_seq, [input_shape[0], input_shape[1], input_shape[2], -1]).permute(0, 1, 3, 2)  # (bs, frame, feature_num, channel)
        self.adj_space = self.adj_space.to(d_seq.device)
        adj_dseq = self.adj_space
        adj_dseq_hat = self.normalize_adj(adj_dseq, add_eye=False)

        d_seq = self.intra_GCN_layer1(d_seq, adj_dseq_hat)  # (bs, frame, feature_num, channel)

        d_seq = d_seq.permute(0, 2, 1, 3)   # (bs, feature_num, frame, channel)
        self.adj_frame = self.adj_frame.to(d_seq.device)
        adj_frame_hat = self.normalize_adj(self.adj_frame, add_eye=True)
        d_seq = self.inter_GCN_layer1(d_seq, adj_frame_hat)
        
        d_seq = d_seq.permute(0, 2, 3, 1)   # (bs, frame, channel, feature_num)
        d_seq = d_seq.reshape(input_shape)  # (bs, frame, channel, H, W, D)
        return d_seq    


# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     bs = 1
#     frame_num = 7
#     channel = 64
#     H, W, D = 5, 16, 16
#     input = torch.rand((bs, frame_num, channel, H, W, D)).to(device)
#     net = DifferentialGCNBlock(frame_num=frame_num,
#                                 feature_shape=[H, W, D], 
#                                 channel_dim=channel, 
#                                 hidden_dim=32,
#                                 dropout=0.6,
#                                 topk=10).to(device)
#     output = net(input)
#     print()
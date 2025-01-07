import copy
from this import d
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,  stride=1, padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class DimensionTransformConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, feature_dim, feature_dim_nonuniform):
        super().__init__()
        '''
        (feature_dim + p - k) // s + 1 = feature_dim_nonuniform
        '''
        if feature_dim == 16 and feature_dim_nonuniform == 5:
            self.block = nn.Conv3d(in_planes, out_planes, kernel_size=(4, 3, 3), stride=(3, 1, 1), padding=(0, 1, 1))
        elif feature_dim == 12 and feature_dim_nonuniform == 5:
            self.block = nn.Conv3d(in_planes, out_planes, kernel_size=(4, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1))
        else:
            self.block = None
        assert self.block is not None

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.InstanceNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.InstanceNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class ChannelWiseMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, dropout, num_heads=12):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(self.all_head_size, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)           # (bs, channel, frame_num, num_head, head_size)
        return x.permute(0, 1, 3, 2, 4)    # (bs, channel, num_head, frame_num, head_size)

    def forward(self, f_prime_seq, d_prime_seq):
        '''
            input: (bs, channel, frame_num, feature_dim*feature_dim*feature_dim)
            output: (bs, channel, frame_num, feature_dim*feature_dim*feature_dim)
        '''
        mixed_query_layer = self.query(d_prime_seq)
        mixed_key_layer = self.key(f_prime_seq)
        mixed_value_layer = self.value(f_prime_seq)
        query_layer = self.transpose_for_scores(mixed_query_layer)  # (bs, channel, num_head, frame_num, head_size)
        key_layer = self.transpose_for_scores(mixed_key_layer)      # (bs, channel, num_head, frame_num, head_size)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (bs, channel, num_head, frame_num, head_size)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 1, 3, 2, 4).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class GADBlock(nn.Module):
    def __init__(self, embed_dim, dropout, num_heads=12):
        super().__init__()
        self.cross_att = ChannelWiseMultiHeadAttention(embed_dim=embed_dim, dropout=dropout, num_heads=num_heads)
        self.self_att = ChannelWiseMultiHeadAttention(embed_dim=embed_dim, dropout=dropout, num_heads=num_heads)
    
    def forward(self, f_prime_seq, d_prime_seq):
        f_prime_seq_att = self.cross_att(f_prime_seq, d_prime_seq)
        d_prime_seq_att = self.self_att(d_prime_seq, d_prime_seq)
        return f_prime_seq_att, d_prime_seq_att


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=786, d_ff=2048, dropout=0.1):
        super().__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, frame_num, embed_dim, dropout):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1, frame_num, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
            input: (bs, frame_num, channel, feature_dim*feature_dim*feature_dim)
            output: (bs, channel, frame_num, feature_dim*feature_dim*feature_dim)
        '''
        x = x.transpose(1, 2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.attention_norm_f = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attention_norm_d = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = GADBlock(embed_dim, dropout)

        self.mlp_norm_f = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_f = PositionwiseFeedForward(embed_dim, 2048)

        self.mlp_norm_d = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_d = PositionwiseFeedForward(embed_dim, 2048)


    def forward(self, f_prime_seq, d_prime_seq):
        '''
            input: (bs, channel, frame_num, feature_dim*feature_dim*feature_dim)
            output: (bs, channel, frame_num, feature_dim*feature_dim*feature_dim)
        '''
        h_f_seq = f_prime_seq
        h_d_seq = d_prime_seq
        f_prime_seq = self.attention_norm_f(f_prime_seq)
        d_prime_seq = self.attention_norm_d(d_prime_seq)
        f_prime_seq, d_prime_seq = self.attn(f_prime_seq, d_prime_seq)

        f_prime_seq = f_prime_seq + h_f_seq
        h_f_seq = f_prime_seq
        f_prime_seq = self.mlp_norm_f(f_prime_seq)
        f_prime_seq = self.mlp_f(f_prime_seq)
        f_prime_seq = f_prime_seq + h_f_seq

        d_prime_seq = d_prime_seq + h_d_seq
        h_d_seq = d_prime_seq
        d_prime_seq = self.mlp_norm_d(d_prime_seq)
        d_prime_seq = self.mlp_d(d_prime_seq)
        d_prime_seq = d_prime_seq + h_d_seq

        return f_prime_seq, d_prime_seq


class Transformer(nn.Module):
    def __init__(self, frame_num, embed_dim, num_layers, dropout, extract_layers):
        super().__init__()
        self.embeddings_f = Embeddings(frame_num, embed_dim, dropout)
        self.embeddings_d = Embeddings(frame_num, embed_dim, dropout)
        self.extract_layers = extract_layers
        self.layer = nn.ModuleList()
        for _ in range(num_layers):
            layer = TransformerBlock(embed_dim, dropout)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, f_prime_seq, d_prime_seq):
        '''
            input: (bs, frame_num, channel, feature_dim*feature_dim*feature_dim)
            output: [(bs, frame_num, channel, feature_dim*feature_dim*feature_dim),...]
        '''
        extract_f_layers = []
        extract_d_layers = []

        f_prime_seq = self.embeddings_f(f_prime_seq)    # (bs, channel, frame_num, feature_dim*feature_dim*feature_dim)
        d_prime_seq = self.embeddings_d(d_prime_seq)    # (bs, channel, frame_num, feature_dim*feature_dim*feature_dim)

        for depth, layer_block in enumerate(self.layer):
            f_prime_seq, d_prime_seq = layer_block(f_prime_seq, d_prime_seq)
            if depth + 1 in self.extract_layers:
                extract_f_layers.append(f_prime_seq.transpose(1, 2))
                extract_d_layers.append(d_prime_seq.transpose(1, 2))
        
        return extract_f_layers, extract_d_layers


class AdaptiveAnatomicKineticFusion(nn.Module):
    """
    compute Adaptive Attention Dual-Modal Fusion 
    """

    def __init__(self, dk):
        super(AdaptiveAnatomicKineticFusion, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.linear1 =  nn.Linear(dk, dk)
        self.linear2 =  nn.Linear(dk, dk, bias=False)

    def forward(self, feature_f, feature_d):
        mu_ct = self.linear2(torch.tanh(self.linear1(feature_f))) # batch*dk
        mu_wsi = self.linear2(torch.tanh(self.linear1(feature_d))) # batch*dk
        mu_combine = torch.stack((mu_ct, mu_wsi), dim=-1) # batch*dk*2
        weight = self.softmax(mu_combine)
        weighted_sum_score = feature_f*weight[..., 0] + feature_d+weight[..., 1]

        return weighted_sum_score


class GADModule(nn.Module):
    '''
    8   [2, 4, 6, 8]
    9   [2, 4, 6, 9]
    10   [2, 4, 7, 10]
    11   [2, 5, 8, 11]
    12   [3, 6, 9, 12]
    13   [3, 6, 9, 13]
    14   [3, 6, 10, 14]
    15   [3, 7, 11, 15]
    16   [4, 8, 12, 16]
    '''
    def __init__(self, frame_num, input_channel, feature_dim, output_channel, num_layers=12, dropout=0.1, extract_layers=[3, 6, 9, 12]):
        super().__init__()
        self.frame_num = frame_num
        self.input_channel = input_channel
        self.feature_dim = feature_dim
        self.embed_dim = feature_dim[0] * feature_dim[1] * feature_dim[2]
        # Transformer Encoder
        self.transformer = Transformer(self.frame_num, self.embed_dim, num_layers, dropout, extract_layers)

        self.fusion = AdaptiveAnatomicKineticFusion(self.embed_dim)

        # U-Net Decoder
        self.decoder0 = nn.Sequential(
                            Deconv3DBlock(input_channel, 256),
                            Deconv3DBlock(256, 128),
                            Deconv3DBlock(128, 64),
                        )

        self.decoder3 = nn.Sequential(
                            Deconv3DBlock(input_channel, 512),
                            Deconv3DBlock(512, 256),
                            Deconv3DBlock(256, 128)
                        )
        self.decoder6 = nn.Sequential(
                            Deconv3DBlock(input_channel, 512),
                            Deconv3DBlock(512, 256),
                        )

        self.decoder9 = Deconv3DBlock(input_channel, 512)

        self.decoder12_upsampler = SingleDeconv3DBlock(input_channel, 512)

        self.decoder9_upsampler =   nn.Sequential(
                                        Conv3DBlock(1024, 512),
                                        Conv3DBlock(512, 512),
                                        Conv3DBlock(512, 512),
                                        SingleDeconv3DBlock(512, 256)
                                    )
        
        self.decoder6_upsampler =   nn.Sequential(
                                        Conv3DBlock(512, 256),
                                        Conv3DBlock(256, 256),
                                        SingleDeconv3DBlock(256, 128)
                                    )

        self.decoder3_upsampler =   nn.Sequential(
                                        Conv3DBlock(256, 128),
                                        Conv3DBlock(128, 128),
                                        Conv3DBlock(128, 64)
                                    )

        self.decoder0_header = nn.Sequential(
                Conv3DBlock(128, 64),
                Conv3DBlock(64, 64),
                SingleConv3DBlock(64, output_channel, 1)
            )

       
    def forward(self, f_prime_seq, d_prime_seq):
        '''
            input: (bs, frame_num, channel, img_shape[0]*feature_dim*feature_dim)
        '''
        z_f, z_d = self.transformer(f_prime_seq, d_prime_seq)

        z0_f, z3_f, z6_f, z9_f, z12_f = f_prime_seq, *z_f
        z0_d, z3_d, z6_d, z9_d, z12_d = d_prime_seq, *z_d

        B, T, C, D = z0_f.size()

        z0_f = z0_f.reshape(-1, D)
        z0_d = z0_d.reshape(-1, D)
        z0 = self.fusion(z0_f, z0_d)
        z0 = z0.reshape(B, T, C, D)

        z3_f = z3_f.reshape(-1, D)
        z3_d = z3_d.reshape(-1, D)
        z3 = self.fusion(z3_f, z3_d)
        z3 = z3.reshape(B, T, C, D)

        z6_f = z6_f.reshape(-1, D)
        z6_d = z6_d.reshape(-1, D)
        z6 = self.fusion(z6_f, z6_d)
        z6 = z6.reshape(B, T, C, D)

        z9_f = z9_f.reshape(-1, D)
        z9_d = z9_d.reshape(-1, D)
        z9 = self.fusion(z9_f, z9_d)
        z9 = z9.reshape(B, T, C, D)

        z12_f = z12_f.reshape(-1, D)
        z12_d = z12_d.reshape(-1, D)
        z12 = self.fusion(z12_f, z12_d)
        z12 = z12.reshape(B, T, C, D)

        z0 = torch.mean(z0, dim=1, keepdim=False).reshape(-1, self.input_channel, self.feature_dim[0], self.feature_dim[1], self.feature_dim[2])
        z3 = torch.mean(z3, dim=1, keepdim=False).reshape(-1, self.input_channel, self.feature_dim[0], self.feature_dim[1], self.feature_dim[2])
        z6 = torch.mean(z6, dim=1, keepdim=False).reshape(-1, self.input_channel, self.feature_dim[0], self.feature_dim[1], self.feature_dim[2])
        z9 = torch.mean(z9, dim=1, keepdim=False).reshape(-1, self.input_channel, self.feature_dim[0], self.feature_dim[1], self.feature_dim[2])
        z12 = torch.mean(z12, dim=1, keepdim=False).reshape(-1, self.input_channel, self.feature_dim[0], self.feature_dim[1], self.feature_dim[2])

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        return output


# if __name__ == '__main__':
#     bath_size = 1
#     # feature_dim = 8
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # net = AdaptiveAttentionFusion(feature_dim).to(device)
#     # enhanced_feature_CT = torch.rand([bath_size, feature_dim]).to(device)
#     # enhanced_feature_WSI = torch.rand([bath_size, feature_dim]).to(device)
#     # out = net(enhanced_feature_CT, enhanced_feature_WSI)
#     # print(out.size())

#     frame_num = 7
#     input_channel = 16
#     feature_dim = 8
#     img_shape = [input_channel, 5*feature_dim*feature_dim]
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net = GADModule(frame_num=frame_num, input_channel=input_channel, feature_dim=[5, feature_dim, feature_dim], output_channel=3).to(device)
#     f_prime_seq = torch.rand([bath_size, frame_num] + img_shape).to(device)
#     d_prime_seq = torch.rand([bath_size, frame_num] + img_shape).to(device)
#     out = net(f_prime_seq, d_prime_seq)
#     print(out.size())

    

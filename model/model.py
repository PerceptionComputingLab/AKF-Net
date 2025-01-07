import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from model.guided_attention_disentangling import GADModule
from model.graph_learning import DRGLBlock

class SingleConv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.block = nn.Conv3d(in_channels, out_channels, kernel_size=(3, kernel_size, kernel_size), 
                                stride=(1, stride, stride), padding=(1, (kernel_size - 1) // 2, (kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class DimensionTransformDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, feature_dim):
        super().__init__()
        feature_dim_0 = feature_dim[0]
        feature_dim_uniform = feature_dim[-1]
        '''
        (feature_dim_0 - 1) x stride + kernel = feature_dim_uniform
        '''
        if feature_dim_0 == 5 and feature_dim_uniform == 16:
            self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=(4, 1, 1), stride=(3, 1, 1), padding=0, output_padding=0)
        elif feature_dim_0 == 5 and feature_dim_uniform == 12:
            self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=(4, 1, 1), stride=(2, 1, 1), padding=0, output_padding=0)
        else:
            self.block = None
        assert self.block is not None

    def forward(self, x):
        return self.block(x)


class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = SingleConv3DBlock(in_channels, out_channels, kernel_size=3, stride=stride)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SingleConv3DBlock(out_channels, out_channels, kernel_size=3, stride=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        if use_1x1conv:
            self.downsample = nn.Sequential(
                SingleConv3DBlock(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.InstanceNorm3d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):
    def __init__(self, image_channel, img_shape, block, layers=[2, 2, 2, 2], layer_out_channels=[64, 128, 256, 512]):
        super(ResNet3D, self).__init__()
        self.layer_in_channels = layer_out_channels[0]
        self.conv1 = SingleConv3DBlock(image_channel, self.layer_in_channels, kernel_size=3, stride=1)
        self.norm1 = nn.InstanceNorm3d(self.layer_in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, layer_out_channels[0], layers[0], first_block=True)
        self.layer2 = self.make_layer(block, layer_out_channels[1], layers[1])
        self.layer3 = self.make_layer(block, layer_out_channels[2], layers[2])
        self.layer4 = self.make_layer(block, layer_out_channels[3], layers[3])

    def make_layer(self, block, out_channels, block_num, first_block=False):
        layers = []
        if first_block:
            layers.append(block(self.layer_in_channels, out_channels, use_1x1conv=False, stride=1))
        else:
            layers.append(block(self.layer_in_channels, out_channels, use_1x1conv=True, stride=2))
        self.layer_in_channels = out_channels
        for _ in range(1, block_num):
            layers.append(block(out_channels, out_channels, use_1x1conv=False, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


def ResNet3D18(image_channel, image_shape, block, layers=[2, 2, 2, 2], layer_out_channels=[64, 128, 256, 512]):
    return ResNet3D(image_channel=image_channel, img_shape=image_shape,  block=block, 
                    layers=layers,  layer_out_channels=layer_out_channels)


def ResNet3D34(image_channel, image_shape, block, layers=[3, 4, 6, 8], layer_out_channels=[64, 128, 256, 512]):
    return ResNet3D(image_channel=image_channel, img_shape=image_shape,  block=block, 
                    layers=layers,  layer_out_channels=layer_out_channels)


class DifferentialGCN(nn.Module):
    def __init__(self, in_features, embed_dim, out_features, bias=True):
        super(DifferentialGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.in_proj = nn.Linear(in_features, embed_dim)
        self.out_proj = nn.Linear(embed_dim, out_features)
        self.weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim).float())
        if bias:
            self.bias = nn.Parameter(torch.Tensor(embed_dim).float())
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        std = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, adj, x):
        x = self.in_proj(x)
        y = torch.matmul(x.float(), self.weight.float())
        output = torch.matmul(adj.float(), y.float())
        if self.bias is not None:
            output += self.bias.float()
        output = self.out_proj(output)
        return output


class ASDBBlcok(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.attention_head_size = embed_dim
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.weight = nn.Linear(embed_dim, embed_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_states1, hidden_states2):
        query_layer = hidden_states1
        key_layer = hidden_states2
        value_layer = hidden_states2

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = self.weight(context_layer)
        attention_output = self.proj_dropout(context_layer)
        return attention_output


class AEPformer(nn.Module):
    def __init__(self, frame_num, img_shape, output_channel, resnet_depth=18, resnet_out_channels=[4, 8, 16, 32], dropout=0.1, GLBlock_num=2):
        super().__init__()
        self.frame_num = frame_num
        if resnet_depth == 18:
            self.img_encoder = ResNet3D18(image_channel=1, image_shape=img_shape, block=BasicBlock3D, layer_out_channels=resnet_out_channels)
            self.dif_encoder = ResNet3D18(image_channel=1, image_shape=img_shape, block=BasicBlock3D, layer_out_channels=resnet_out_channels)
        elif resnet_depth == 34:
            self.img_encoder = ResNet3D34(image_channel=1, image_shape=img_shape, block=BasicBlock3D, layer_out_channels=resnet_out_channels)
            self.dif_encoder = ResNet3D34(image_channel=1, image_shape=img_shape, block=BasicBlock3D, layer_out_channels=resnet_out_channels)
        else:
            self.img_encoder, self.dif_encoder = None, None
        assert self.img_encoder is not None
        assert self.dif_encoder is not None
        self.feature_dim = img_shape[-1] // 8
        self.feature_shape = [resnet_out_channels[3], img_shape[0], self.feature_dim, self.feature_dim] # [channel, img_shape[0], feature_dim, feature_dim]
        DRGL_model  = []
        for _ in range(GLBlock_num):
            DRGL_model.append(DRGLBlock(frame_num=self.frame_num-1,
                                                feature_shape=self.feature_shape[1:],
                                                channel_dim=self.feature_shape[0],
                                                hidden_dim=32,
                                                dropout=dropout))
        self.DRGL = nn.Sequential(*DRGL_model)
        
        
        self.ASDB = ASDBBlcok(
                            embed_dim=img_shape[0]*self.feature_dim*self.feature_dim, 
                            dropout=dropout
                        )
        self.GAD =    GADModule(
                            frame_num=self.frame_num,
                            input_channel=resnet_out_channels[3],
                            feature_dim=[img_shape[0], self.feature_dim, self.feature_dim],
                            output_channel=output_channel,
                            dropout=dropout
                        )

    def forward(self, image):
        img_shape = torch.tensor(image.size()).tolist() # [bs, frame_num, image_dim, image_dim, image_dim]
        batch_size = img_shape[0]
        frame_num = img_shape[1]
        assert frame_num == self.frame_num
        img_size = img_shape[2:] # [image_dim, image_dim, image_dim]
        
        # D(t-1, t) = It - It-1
        difference = torch.zeros([batch_size, frame_num-1] + img_size).to(image.device)
        for i in range(0, frame_num - 1):
            difference[:, i, :, :, :] = F.layer_norm(image[:, i + 1, :, :, :] - image[:, i, :, :, :], normalized_shape=img_size)
        
        # Encode Image
        f_seq = []
        for i in range(0, frame_num):
            I_i = image[:, i, :, :, :].unsqueeze(1)
            f_i = self.img_encoder(I_i)
            f_seq.append(F.layer_norm(f_i, normalized_shape=f_i.shape[1:]).unsqueeze(1))
        f_seq = torch.concat(f_seq, dim=1)  # (bs, frame_num, channel, img_shape[0], feature_dim, feature_dim)
        assert self.feature_shape == torch.tensor(f_seq.size()[2:]).tolist()
        f_seq = torch.reshape(f_seq, (batch_size, frame_num, self.feature_shape[0], -1))    # (bs, frame_num, channel, img_shape[0]*feature_dim*feature_dim)
        
        # Encode Difference
        d_seq = []
        for i in range(0, frame_num - 1):
            D_i = difference[:, i, :, :, :].unsqueeze(1)
            d_i = self.dif_encoder(D_i)
            d_seq.append(F.layer_norm(d_i, normalized_shape=d_i.shape[1:]).unsqueeze(1))
        d_seq = torch.concat(d_seq, dim=1)  # (bs, frame_num-1, channel, img_shape[0], feature_dim, feature_dim)

        # Dynamic Recycle Graph Learning
        d_prime_seq = self.DRGL(d_seq)    # (bs, frame_num-1, channel, img_shape[0], feature_dim, feature_dim)

        d_prime_seq = torch.reshape(d_prime_seq, (batch_size, frame_num-1, self.feature_shape[0], -1))

        f_seq_0 = f_seq[:, 0, :, :]
        f_hat_seq = [f_seq_0.unsqueeze(1)]
        for i in range(1, frame_num):
            d_prim_sum = torch.sum(d_prime_seq[:, :i, :, :], dim=1, keepdim=False)
            f_hat_i  = F.layer_norm(f_seq_0 + d_prim_sum, normalized_shape=f_seq_0.shape[1:])
            f_hat_seq.append(f_hat_i.unsqueeze(1))  
        f_hat_seq = torch.concat(f_hat_seq, dim=1)     # (bs, frame_num, channel, img_shape[0]*feature_dim*feature_dim)

        # Adaptive Static-Dynamic Blending
        f_prime_seq = []
        for i in range(0, frame_num):
            f_seq_i = f_seq[:, i, :, :]
            f_hat_i = f_hat_seq[:, i, :, :]
            f_prime_i = self.ASDB(f_hat_i, f_seq_i)
            f_prime_seq.append(f_prime_i.unsqueeze(1))
        f_prime_seq = torch.concat(f_prime_seq, dim=1)  # (bs, frame_num, channel, img_shape[0]*feature_dim*feature_dim)
        d_prime_seq_cat_f_prime_0 = torch.concat([f_prime_seq[:, 0, :, :].unsqueeze(1), d_prime_seq], dim=1)    # (bs, frame_num, channel, img_shape[0]*feature_dim*feature_dim)
        
        out = self.GAD(f_prime_seq, d_prime_seq_cat_f_prime_0)
       
        return out

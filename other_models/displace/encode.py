import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from FNet import FNet


def get_neighbor_index(vertices: "(bs, vertice_num, 3)", neighbor_num: int):
    """
    Return: (bs, vertice_num, neighbor_num)
    """
    bs, v, _ = vertices.size()
    device = vertices.device
    inner = torch.bmm(vertices, vertices.transpose(1, 2))  # (bs, v, v)
    quadratic = torch.sum(vertices**2, dim=2)  # (bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    neighbor_index = torch.topk(
        distance, k=neighbor_num + 1, dim=-1, largest=False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    return neighbor_index


def get_nearest_index(target: "(bs, v1, 3)", source: "(bs, v2, 3)"):
    """
    Return: (bs, v1, 1)
    """
    inner = torch.bmm(target, source.transpose(1, 2))  # (bs, v1, v2)
    s_norm_2 = torch.sum(source**2, dim=2)  # (bs, v2)
    t_norm_2 = torch.sum(target**2, dim=2)  # (bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(2) - 2 * inner
    nearest_index = torch.topk(d_norm_2, k=1, dim=-1, largest=False)[1]
    return nearest_index


def indexing_neighbor(tensor: "(bs, vertice_num, dim)",
                      index: "(bs, vertice_num, neighbor_num)"):
    """
    Return: (bs, vertice_num, neighbor_num, dim)
    """
    bs, v, n = index.size()
    id_0 = torch.arange(bs).view(-1, 1, 1)
    tensor_indexed = tensor[id_0, index]
    return tensor_indexed


def get_neighbor_displacement(
        vertices: "(bs, vertice_num, 3)",
        neighbor_index: "(bs, vertice_num, neighbor_num)"):
    """
    Return: (bs, vertice_num, neighobr_num, 3)
    """
    neighbors = indexing_neighbor(vertices, neighbor_index)  # (bs, v, n, 3)
    neighbor_anchored = neighbors - vertices.unsqueeze(2)
    return neighbor_anchored


class Operator3D(nn.Module):
    """
    Extract structure feafure from surface, independent from vertice coordinates
    """
    def __init__(self, kernel_num, support_num):
        super().__init__()
        self.kernel_num = kernel_num
        self.support_num = support_num

        self.relu = nn.ReLU(inplace=True)
        self.weights = nn.Parameter(
            torch.FloatTensor(1, 1, support_num, kernel_num))
        self.displacement = nn.Parameter(
            torch.FloatTensor(3, support_num * kernel_num))
        # self.fourier_map = Periodics(dim_input=3, dim_output=32)
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.support_num * self.kernel_num)
        self.weights.data.uniform_(-stdv, stdv)
        self.displacement.data.uniform_(-stdv, stdv)

    def forward(self, neighbor_index: "(bs, vertice_num, neighbor_num)",
                vertices: "(bs, vertice_num, 3)"):
        """
        Return vertices with local feature: (bs, vertice_num, kernel_num)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_displacement = get_neighbor_displacement(
            vertices, neighbor_index)

        # NOTE displacements are not normalized
        # with shape of (bs, vertice_num, neighbor_num, s*k)
        theta = neighbor_displacement @ self.displacement  

        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num,
                                        self.support_num, self.kernel_num)
        theta = torch.max(
            theta, dim=2)[0] * self.weights
        # (bs, vertice_num, support_num, kernel_num)
        feature = torch.sum(theta, dim=2)
        # (bs, vertice_num, kernel_num)
        # freq = self.fnet_map(vertices)
        # return torch.cat((feature, freq), 2)
        return feature

class OperatorND(nn.Module):
    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        # arguments:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.support_num = support_num

        # parameters:
        self.relu = nn.ReLU(inplace=True)
        self.weights = nn.Parameter(
            torch.FloatTensor(in_channel, (support_num + 1) * out_channel))
        self.bias = nn.Parameter(
            torch.FloatTensor((support_num + 1) * out_channel))
        self.displacement = nn.Parameter(
            torch.FloatTensor(3, support_num * out_channel))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.out_channel * (self.support_num + 1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.displacement.data.uniform_(-stdv, stdv)

    def forward(self, neighbor_index: "(bs, vertice_num, neighbor_index)",
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, in_channel)"):
        """
        Return: output feature map: (bs, vertice_num, out_channel)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_displacement = get_neighbor_displacement(
            vertices, neighbor_index)
        theta = neighbor_displacement @ self.displacement
        # (bs, vertice_num, neighbor_num, support_num * out_channel)
        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, -1)
        # (bs, vertice_num, neighbor_num, support_num * out_channel)

        feature_out = feature_map @ self.weights + self.bias
        # (bs, vertice_num, (support_num + 1) * out_channel)
        feature_center = feature_out[:, :, :self.out_channel]
        # (bs, vertice_num, out_channel)
        feature_support = feature_out[:, :, self.out_channel:]
        # (bs, vertice_num, support_num * out_channel)

        # Fuse together - max among product
        feature_support = indexing_neighbor(
            feature_support, neighbor_index
        )  
        # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = theta * feature_support  
        # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = activation_support.view(
            bs, vertice_num, neighbor_num, self.support_num, self.out_channel)
        activation_support = torch.max(
            activation_support,
            dim=2)[0]  
        # (bs, vertice_num, support_num, out_channel)
        activation_support = torch.sum(
            activation_support, dim=2)  # (bs, vertice_num, out_channel)
        feature_fuse = feature_center + activation_support  
        # (bs, vertice_num, out_channel)
        return feature_fuse

class Pooling(nn.Module):
    def __init__(self, pooling_rate: int = 4, neighbor_num: int = 4):
        super().__init__()
        self.pooling_rate = pooling_rate
        self.neighbor_num = neighbor_num

    def forward(self, vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, channel_num)"):
        """
        Return:
            vertices_pool: (bs, pool_vertice_num, 3),
            feature_map_pool: (bs, pool_vertice_num, channel_num)
        """
        bs, vertice_num, _ = vertices.size()
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        neighbor_feature = indexing_neighbor(
            feature_map,
            neighbor_index)  
        # (bs, vertice_num, neighbor_num, channel_num)
        pooled_feature = torch.max(
            neighbor_feature, dim=2)[0]  
        # (bs, vertice_num, channel_num)

        pool_num = int(vertice_num / self.pooling_rate)
        sample_idx = torch.randperm(vertice_num)[:pool_num]
        vertices_pool = vertices[:, sample_idx, :]  
        # (bs, pool_num, 3)
        feature_map_pool = pooled_feature[:, sample_idx, :]  
        # (bs, pool_num, channel_num)
        return vertices_pool, feature_map_pool


class Encoder(nn.Module):
    def __init__(self, support_num: int, neighbor_num: int):
        super().__init__()
        self.neighbor_num = neighbor_num

        self.conv_0 = Operator3D(kernel_num=32, support_num=support_num)
        self.dropping = torch.nn.Dropout(p=0.1, inplace=True)
        self.conv_1 = OperatorND(32, 64, support_num=support_num)
        self.pool_1 = Pooling(pooling_rate=8, neighbor_num=8)
        self.conv_2 = OperatorND(64, 128, support_num=support_num)
        self.conv_3 = OperatorND(128, 256, support_num=support_num)
        self.pool_2 = Pooling(pooling_rate=8, neighbor_num=8)
        self.conv_4 = OperatorND(256, 512, support_num=support_num)
        self.conv_5 = OperatorND(512, 512, support_num=support_num)
        self.pool_3 = Pooling(pooling_rate=8, neighbor_num=8)
        self.conv_6 = OperatorND(512, 1024, support_num=support_num)

    def forward(self, vertices: "(bs, vertice_num, 3)"):
        bs, vertice_num, _ = vertices.size()

        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        fm_0 = self.conv_0(neighbor_index, vertices)
        fm_0 = self.dropping(fm_0)
        fm_0 = F.relu(fm_0, inplace=True)
        fm_1 = self.conv_1(neighbor_index, vertices, fm_0)
        fm_1 = F.relu(fm_1, inplace=True)
        vertices, fm_1 = self.pool_1(vertices, fm_1)
        vertices_anchor = vertices
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)

        fm_2 = self.conv_2(neighbor_index, vertices, fm_1)
        fm_2 = F.relu(fm_2, inplace=True)
        fm_3 = self.conv_3(neighbor_index, vertices, fm_2)
        fm_3 = F.relu(fm_3, inplace=True)
        vertices, fm_3 = self.pool_2(vertices, fm_3)
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)

        fm_4 = self.conv_4(neighbor_index, vertices, fm_3)
        fm_4 = F.relu(fm_4, inplace=True)
        fm_5 = self.conv_5(neighbor_index, vertices, fm_4)
        fm_5 = F.relu(fm_5, inplace=True)
        vertices, fm_5 = self.pool_3(vertices, fm_5)
        neighbor_index = get_neighbor_index(vertices, 3)

        fm_6 = self.conv_6(neighbor_index, vertices, fm_5)
        feature_global = fm_6.max(1, keepdim=True)[0]
        return feature_global, vertices_anchor

class Disp3D(nn.Module):
    def __init__(self, class_num, support_num, neighbor_num):
        super().__init__()
        self.neighbor_num = neighbor_num

        self.conv_0 = Operator3D(kernel_num=128, support_num=support_num)
        self.conv_1 = OperatorND(128, 128, support_num=support_num)
        self.pool_1 = Pooling(pooling_rate=4, neighbor_num=4)
        self.conv_2 = OperatorND(128, 256, support_num=support_num)
        self.conv_3 = OperatorND(256, 256, support_num=support_num)
        self.pool_2 = Pooling(pooling_rate=4, neighbor_num=4)
        self.conv_4 = OperatorND(256, 512, support_num=support_num)

        dim_fuse = sum([128, 128, 256, 256, 512, 512])
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(dim_fuse, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, class_num, 1),
        )

    def forward(self, 
                vertices: "tensor (bs, vetice_num, 3)"):
        """
        Return: (bs, vertice_num, class_num)
        """

        bs, vertice_num, _ = vertices.size()
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)

        fm_0 = F.relu(self.conv_0(neighbor_index, vertices), inplace=True)
        fm_1 = F.relu(self.conv_1(neighbor_index, vertices, fm_0), inplace=True)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        neighbor_index = get_neighbor_index(v_pool_1, self.neighbor_num)

        fm_2 = F.relu(self.conv_2(neighbor_index, v_pool_1, fm_pool_1), inplace=True)
        fm_3 = F.relu(self.conv_3(neighbor_index, v_pool_1, fm_2), inplace=True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        neighbor_index = get_neighbor_index(v_pool_2, self.neighbor_num)

        fm_4 = self.conv_4(neighbor_index, v_pool_2, fm_pool_2)
        f_global = fm_4.max(1)[0] #(bs, f)

        nearest_pool_1 = get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = get_nearest_index(vertices, v_pool_2)
        fm_2 = indexing_neighbor(fm_2, nearest_pool_1).squeeze(2)
        fm_3 = indexing_neighbor(fm_3, nearest_pool_1).squeeze(2)
        fm_4 = indexing_neighbor(fm_4, nearest_pool_2).squeeze(2)
        f_global = f_global.unsqueeze(1).repeat(1, vertice_num, 1)
        # (bs, vertice_num, cat_one_hot)
        fm_fuse = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4, f_global], dim=2)

        conv1d_input = fm_fuse.permute(0, 2, 1) # (bs, fuse_ch, vertice_num)
        conv1d_out = self.conv1d_block(conv1d_input) 
        pred = conv1d_out.permute(0, 2, 1) # (bs, vertice_num, ch)
        return pred

def test():
    import time
    bs = 8
    v = 1024
    dim = 3
    n = 20
    vertices = torch.randn(bs, v, dim)
    neighbor_index = get_neighbor_index(vertices, n)

    s = 3
    conv_1 = Operator3D(kernel_num=32, support_num=s)
    conv_2 = OperatorND(in_channel=32, out_channel=64, support_num=s)
    pool = Pooling(pooling_rate=4, neighbor_num=4)

    print("Input size: {}".format(vertices.size()))
    start = time.time()
    f1 = conv_1(neighbor_index, vertices)
    print("\n[1] Time: {}".format(time.time() - start))
    print("[1] Out shape: {}".format(f1.size()))
    start = time.time()
    f2 = conv_2(neighbor_index, vertices, f1)
    print("\n[2] Time: {}".format(time.time() - start))
    print("[2] Out shape: {}".format(f2.size()))
    start = time.time()
    v_pool, f_pool = pool(vertices, f2)
    print("\n[3] Time: {}".format(time.time() - start))
    print("[3] v shape: {}, f shape: {}".format(v_pool.size(), f_pool.size()))


if __name__ == "__main__":
    test()

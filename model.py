import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from itertools import combinations
from torch.nn.init import xavier_normal_ 

from torch.nn.modules.activation import MultiheadAttention

from torch.autograd import Variable
import torchvision.models as models
from utils import extract_class_indices, cos_sim
from einops import rearrange


class CNN_FSHead(nn.Module):
    """
    Base class which handles a few-shot method. Contains a CLIP vision backbone.
    """

    def __init__(self, args):
        super(CNN_FSHead, self).__init__()
        self.train()
        self.args = args

        # Replace previous backbones with CLIP backbone
        # self.backbone = CLIPBackbone()

        last_layer_idx = -1

        if self.args.backbone == "resnet18":
            backbone = models.resnet18(pretrained=True)
        elif self.args.backbone == "resnet34":
            backbone = models.resnet34(pretrained=True)
        elif self.args.backbone == "resnet50":
            backbone = models.resnet50(pretrained=True)

        self.backbone = nn.Sequential(*list(backbone.children())[:last_layer_idx])

    def get_feats(self, support_images, target_images):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.backbone(support_images)
        target_features = self.backbone(target_images)

        dim = int(support_features.shape[1])

        support_features = support_features.reshape(-1, self.args.seq_len, dim)
        target_features = target_features.reshape(-1, self.args.seq_len, dim)

        return support_features, target_features

    def forward(self, support_images, support_labels, target_images):
        """
        Should return a dict containing logits which are required for computing accuracy. Dict can also contain
        other info needed to compute the loss.
        """
        raise NotImplementedError

    def distribute_model(self):
        """
        Use to split the backbone evenly over all GPUs. Modify if you have other components.
        """
        if self.args.num_gpus > 1:
            self.backbone.cuda(0)
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(0, self.args.num_gpus)])

    def loss(self, task_dict, model_dict):
        """
        Takes in a task dict containing labels etc.
        Takes in the model output dict, which contains "logits", as well as any other info needed to compute the loss.
        Default is cross entropy loss.
        """
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())


class PositionalEncoding(nn.Module):
    """
    Positional encoding from the Transformer paper.
    """
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
       return self.dropout(x)


class MultiTemporalCrossTransformer(nn.Module):
    def __init__(self, args, temporal_set_size=3,  d_model=1152, num_heads=2):
        super(MultiTemporalCrossTransformer, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.args = args
        self.temporal_set_size = temporal_set_size
        self.num_heads = num_heads

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)

        self.class_softmax = torch.nn.Softmax(dim=-1)


        # Generate all tuples
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = nn.ParameterList([nn.Parameter(torch.tensor(comb), requires_grad=False) for comb in frame_combinations])
        self.tuples_len = len(self.tuples)

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.sigma = 1.0

        # 定义线性层
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        x = x.view(x.shape[0], -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def forward(self, support_set, support_labels, queries):
        n_queries = queries.shape[0]
        n_support = support_set.shape[0]

        # Static positional encoding
        support_set = self.pe(support_set)
        queries = self.pe(queries)

        # Construct new queries and support sets made of tuples of images after pe
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]
        support_set = torch.stack(s, dim=-2)
        queries = torch.stack(q, dim=-2)

        # Apply linear maps
        support_set_ks = self.k_linear(support_set)
        queries_ks = self.k_linear(queries)
        support_set_vs = self.v_linear(support_set)
        queries_vs = self.v_linear(queries)

        # Apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks)
        mh_queries_ks = self.norm_k(queries_ks)
        mh_support_set_vs = support_set_vs
        mh_queries_vs = queries_vs

        unique_labels = torch.unique(support_labels)

        # Init tensor to hold distances between every support tuple and every target tuple
        all_distances_tensor = torch.zeros(n_queries, self.args.way, device=queries.device)

        for label_idx, c in enumerate(unique_labels):
            # Select keys and values for just this class
            class_k = torch.index_select(mh_support_set_ks, 0, extract_class_indices(support_labels, c))
            class_v = torch.index_select(mh_support_set_vs, 0, extract_class_indices(support_labels, c))
            k_bs = class_k.shape[0]

            # 分成多头
            query = self.split_heads(mh_queries_ks)
            key = self.split_heads(class_k)
            value = self.split_heads(class_v)

            # 计算多头注意力
            scores = torch.matmul(query.unsqueeze(1), key.transpose(-2, -1)) / torch.sqrt(
                torch.tensor(self.depth, dtype=torch.float32))
            # 对每个头分别应用 softmax
            attention_weights = []
            for head in range(self.num_heads):
                single_head_scores = scores[:, :, head, :, :]  # 取出当前头的分数
                # reshape etc. to apply a softmax for each query tuple
                single_head_scores = single_head_scores.permute(0, 2, 1, 3)
                single_head_scores = single_head_scores.reshape(n_queries, self.tuples_len, -1)
                single_head_scores = [self.class_softmax(single_head_scores[i]) for i in range(n_queries)]
                single_head_scores = torch.cat(single_head_scores)
                single_head_scores = single_head_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len)
                single_head_scores = single_head_scores.permute(0, 2, 1, 3)

                attention_weights.append(single_head_scores)  # 保存每个头的 softmax 结果
            # 合并所有头的 softmax 结果
            attention_weights = torch.stack(attention_weights, dim=2)  # [batch_size, num_heads, seq_len, seq_len]
            # 使用注意力权重加权求和值
            attention_output = torch.matmul(attention_weights, value)  # [batch_size, num_heads, seq_len, depth]
            # 合并多头输出
            query_prototype = attention_output.transpose(2, 3).reshape(n_queries, -1, self.tuples_len, self.d_model)
            query_prototype = torch.sum(query_prototype, dim=1)  # shape: (n_queries, trans_linear_out_dim)
            # 计算距离
            diff = mh_queries_vs - query_prototype
            norm_sq = torch.norm(diff, dim=[-2, -1]) ** 2
            distance = torch.div(norm_sq, self.tuples_len)

            # 将距离乘以 -1 得到 logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:, c_idx] = distance

        return_dict = {'logits': all_distances_tensor}

        return return_dict



class CNN_Multihead(CNN_FSHead):
    """
    Backbone connected to Temporal Cross Transformers of multiple cardinalities.
    """
    def __init__(self, args):
        super(CNN_Multihead, self).__init__(args)
        #fill default args
        self.args.trans_linear_out_dim = 1152
        self.args.temp_set = [2]
        self.args.trans_dropout = 0.1
        self.num_heads = 4
        self.depth = self.args.trans_linear_out_dim // self.num_heads

        self.transformers = nn.ModuleList([MultiTemporalCrossTransformer(args, s, self.args.trans_linear_out_dim, self.num_heads) for s in args.temp_set])

    def forward(self, support_images, support_labels, target_images):
        support_features, target_features = self.get_feats(support_images, target_images)
        all_logits = [t(support_features, support_labels, target_features)['logits'] for t in self.transformers]
        all_logits = torch.stack(all_logits, dim=-1)
        sample_logits = all_logits 
        sample_logits = torch.mean(sample_logits, dim=[-1])

        return_dict = {'logits': sample_logits}
        return return_dict

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs. Leaves TRX on GPU 0.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.backbone.cuda(0)
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.transformers.cuda(0)
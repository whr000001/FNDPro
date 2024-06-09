from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.conv import RGCNConv
from torch_geometric.nn.conv import HGTConv
import torch.nn as nn
import torch
from einops import rearrange, repeat
from torch_scatter import scatter
import math


class UserNode(nn.Module):
    def __init__(self, user_profile_dim, embedding_dim, hidden_dim):
        super().__init__()
        self.property_fc = nn.Linear(user_profile_dim, hidden_dim // 2)
        self.description_fc = nn.Linear(embedding_dim, hidden_dim // 2)

    def forward(self, x):
        properties = self.property_fc(x.profile)
        descriptions = self.description_fc(x.description)
        return torch.cat([properties, descriptions], dim=-1)


class TweetNode(nn.Module):
    def __init__(self, tweet_profile_dim, embedding_dim, hidden_dim):
        super().__init__()
        self.content_fc = nn.Linear(embedding_dim, hidden_dim // 2)
        self.property_fc = nn.Linear(tweet_profile_dim, hidden_dim // 2)

    def forward(self, x):
        contents = self.content_fc(x.content)
        properties = self.property_fc(x.profile)
        return torch.cat([contents, properties], dim=-1)


class SourceNode(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.description_fc = nn.Linear(embedding_dim, hidden_dim)

    def forward(self, x):
        return self.description_fc(x.description)


class NewsNode(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.title_fc = nn.Linear(embedding_dim, hidden_dim // 2)
        self.content_fc = nn.Linear(embedding_dim, hidden_dim // 2)

    def forward(self, x):
        titles = self.title_fc(x.title)
        contents = self.content_fc(x.content)
        return torch.cat([titles, contents], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, heads=4):
        super().__init__()
        dim_head = hidden_dim // heads
        inner_dim = dim * heads
        project_out = not(heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class RelTemporalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=400):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) *
                             -(math.log(10000.0) / hidden_dim))
        emb = nn.Embedding(max_len, hidden_dim)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(hidden_dim)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(hidden_dim)
        emb.requires_grad = False
        self.max_len = max_len
        self.emb = emb
        self.lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, t):
        t = torch.abs(t)
        t = torch.clamp(t, max=self.max_len-1)
        return self.lin(self.emb(t))


class TaHid(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, metadata, mode,
                 num_cls=2, embedding_dim=768, user_profile_dim=18, tweet_profile_dim=5,
                 use_token=True, use_time_embedding=True):
        super().__init__()
        self.linear_dict = nn.ModuleDict()
        meta_tmp = []
        for item in metadata[1]:
            meta_tmp.append(item)
            meta_tmp.append((item[2], 'r{}'.format(item[1]), item[0]))
        metadata = (metadata[0], meta_tmp)
        self.type_index = {item: index for index, item in enumerate(metadata[1])}

        for node_type in metadata[0]:
            if node_type == 'news':
                self.linear_dict[node_type] = NewsNode(embedding_dim=embedding_dim,
                                                       hidden_dim=hidden_dim)
            elif node_type == 'tweet':
                self.linear_dict[node_type] = TweetNode(embedding_dim=embedding_dim,
                                                        tweet_profile_dim=tweet_profile_dim,
                                                        hidden_dim=hidden_dim)
            elif node_type == 'user':
                self.linear_dict[node_type] = UserNode(user_profile_dim=user_profile_dim,
                                                       embedding_dim=embedding_dim,
                                                       hidden_dim=hidden_dim)
            elif node_type == 'source':
                self.linear_dict[node_type] = SourceNode(embedding_dim=embedding_dim,
                                                         hidden_dim=hidden_dim)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            if mode == 'HGT':
                self.convs.append(HGTConv(hidden_dim, hidden_dim, metadata))
            elif mode == 'GCN':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif mode == 'GAT':
                self.convs.append(GATConv(hidden_dim, hidden_dim))
            elif mode == 'RGCN':
                self.convs.append(RGCNConv(hidden_dim, hidden_dim, len(metadata[1])))
        self.attn = Attention(dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.use_token = use_token
        self.use_time_embedding = use_time_embedding
        if self.use_token:
            self.token = nn.Parameter(torch.randn(1, 1, hidden_dim))
            self.pos_embedding = nn.Parameter(torch.randn(1, num_layers + 2, hidden_dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_layers + 1, hidden_dim))
        if self.use_time_embedding:
            self.emd = nn.ModuleDict()
            for item in metadata[1]:
                key = '{}_{}_{}'.format(item[0], item[1], item[2])
                self.emd[key] = RelTemporalEncoding(hidden_dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_cls)
        self.act_fn = nn.ReLU()
        self.metadata = metadata
        self.mode = mode

    def forward(self, x):
        batch_size = x['news'].batch_size
        edge_index_dict = {}
        for key, value in x.edge_index_dict.items():
            src, rel, dst = key
            edge_index_dict[(src, rel, dst)] = value
            edge_index_dict[(dst, 'r{}'.format(rel), src)] = torch.stack([value[1], value[0]])
        x_dict = {
            node_type: self.linear_dict[node_type](x[node_type]).relu_()
            for node_type in x.node_types
        }
        if self.use_time_embedding:
            t_dict = {
                node_type: x[node_type].time
                for node_type in x.node_types
            }
            for edge_type, edge_index in x.edge_index_dict.items():
                src, rel, dst = edge_type
                time_span = t_dict[dst][edge_index[1]] - t_dict[src][edge_index[0]]
                time_span = torch.div(time_span, 60 * 60 * 24 * 30, rounding_mode='trunc')
                time_span = self.emd['{}_{}_{}'.format(src, rel, dst)](time_span)
                time_span = scatter(time_span, edge_index[1], dim=0, dim_size=x_dict[dst].shape[0])
                x_dict[dst] = x_dict[dst] + time_span
        x_dict = {
            key: self.dropout(value)
            for key, value in x_dict.items()
        }
        temporal_feature = [x_dict['news'][:batch_size]]
        if self.mode == 'HGT':
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
                temporal_feature.append(x_dict['news'][:batch_size])
        else:
            offset = {}
            cnt = 0
            feature_cache = []
            for key in self.metadata[0]:
                value = x_dict[key]
                feature_cache.append(value)
                offset[key] = cnt
                cnt += value.shape[0]
            feature = torch.cat(feature_cache, dim=0)
            edge_index_cache = []
            edge_type_cache = []
            for edge_type, edge_index in edge_index_dict.items():
                src_type, _, dst_type = edge_type
                src_index = edge_index[0] + offset[src_type]
                dst_index = edge_index[1] + offset[dst_type]
                index = torch.stack([src_index, dst_index])
                edge_index_cache.append(index)
                edge_type_cache += [self.type_index[edge_type]] * edge_index.shape[1]
            edge_index = torch.cat(edge_index_cache, dim=1)
            edge_type = torch.tensor(edge_type_cache, dtype=torch.long).to(edge_index.device)
            for conv in self.convs:
                if self.mode == 'RGCN':
                    feature = conv(feature, edge_index, edge_type)
                else:
                    feature = conv(feature, edge_index)
                feature = self.dropout(self.act_fn(feature))
                temporal_feature.append(feature[:batch_size])
        temporal_feature = torch.stack(temporal_feature)
        temporal_feature = temporal_feature.transpose(0, 1)
        if self.use_token:
            token = repeat(self.token, '() n d -> b n d', b=batch_size)
            temporal_feature = torch.cat([token, temporal_feature], dim=1)
        _, t, _ = temporal_feature.shape
        temporal_feature += self.pos_embedding[:, :(t+1)]
        temporal_feature = self.attn(temporal_feature)
        if self.use_token:
            temporal_feature = temporal_feature[:, 0]
        else:
            temporal_feature = torch.mean(temporal_feature, dim=1)
        return self.classifier(temporal_feature)

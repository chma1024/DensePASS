"""Dual Attention Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import ResNet50
import math
import functools

from ops.match_boundary.modules.match_boundary import MatchBoundary
from ops.match_class.modules.match_class import MatchClass
from ops.follow_cluster.modules.follow_cluster import FollowCluster
from ops.vcount_cluster.modules.vcount_cluster import VcountCluster
from ops.split_repscore.modules.split_repscore import SplitRepscore

from inplace_abn import InPlaceABN, InPlaceABNSync
BatchNorm2d_relu = functools.partial(InPlaceABN)

class RCB(nn.Module):
    def __init__(self):
        super(RCB, self).__init__()
        self.match_class = MatchClass()
        self.match_boundary = MatchBoundary()
        self.conv_1x1_group = nn.Conv2d(1, 1, 1, bias=True)
        # self.fc_group = nn.Conv2d(1, 1, 1, bias=True)
        nn.init.constant_(self.conv_1x1_group.weight,1)
        nn.init.constant_(self.conv_1x1_group.bias,0)

    def forward(self, semantic_score, boundary_score):
        # compute region attention map
        class_max_prob_A_index = semantic_score.softmax(1).max(1, keepdim=True)[1].int()
        edge_prob = boundary_score.softmax(1)[:, 1:, ...].contiguous()
        semantic_tables = self.match_class(semantic_score.softmax(1), class_max_prob_A_index)  # (bs,9409,9409)
        boundary_tables = self.match_boundary(edge_prob)  # (bs,9409,9409)
        region_attention_tables = (1 - semantic_tables)*(1 - boundary_tables)  # (bs,9409,9409)
        # group process
        region_dicision_tables = (region_attention_tables+region_attention_tables.permute(0,2,1))/2  # (bs,9409,9409)
        return region_attention_tables, region_dicision_tables

class RIB(nn.Module):
    def __init__(self, in_channels=512, out_channels=512,k=8):
        super(RIB, self).__init__()
        self.follow_cluster = FollowCluster(0.8)
        self.vcount_cluster = VcountCluster()
        self.split_repscore = SplitRepscore()
        self.feats_conv = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm2d_relu(512),
            # nn.Dropout2d(0.1),
        )
        self.query_conv = nn.Sequential(nn.Conv2d(512,512, 3, padding=1, bias=False),
                                   BatchNorm2d_relu(512))
        # self.conv_key = nn.Sequential(nn.Conv2d(512,512, 1, bias=False),
        #                                    BatchNorm2d_relu(512))
        self.value_conv = nn.Sequential(nn.Conv2d(512,512, 3, padding=1, bias=False),
                                   BatchNorm2d_relu(512))
        self.final_conv = nn.Sequential(nn.Conv2d(1024,512, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d_relu(512),
                                   nn.Dropout2d(0.1))
        self.collect_conv = nn.Sequential(nn.Conv2d(512,512,3, padding=1,bias=False),BatchNorm2d_relu(512))
        self.interact_conv = nn.Sequential(nn.Conv2d(512,512,3, padding=1,bias=False),BatchNorm2d_relu(512))
        self.distribute_conv = nn.Sequential(nn.Conv2d(512,512,3, padding=1,bias=False),BatchNorm2d_relu(512))
        self.k = k

    def forward(self, feats, region_attention_tables, region_dicision_tables):
        feats = self.feats_conv(feats)
        feats_query = self.query_conv(feats)
        feats_value = self.value_conv(feats)
        region_maps = self.follow_cluster(region_dicision_tables)
        contextual_feats = []
        for bs_idx in range(region_maps.shape[0]):
            region_map = region_maps[bs_idx]
            region_attention_table = region_attention_tables[bs_idx]
            # representative_score = representative_scores[bs_idx]
            feat_query = feats_query[bs_idx]
            feat_value = feats_value[bs_idx]
            # build region image
            with torch.no_grad():
                for i in range(10):
                    region_map = region_map.gather(0, region_map.long())
                cluster_idx = 0
                for cluster_pos in region_map.unique().tolist():
                    region_map = torch.where(region_map == cluster_pos,
                                torch.ones_like(region_map) * (cluster_idx), region_map)
                    cluster_idx +=1

            vcount = self.vcount_cluster(region_attention_table, region_map)  # (num_clusters,9409)
            representative_score = (vcount/((vcount>0).sum(1,keepdim=True).float())).sum(0)
            vtopk_table = vcount.topk(self.k, dim=1).indices.long().reshape(-1).unique()
            reshaped_feat_query = feat_query.reshape((feat_query.shape[0], -1))
            reshaped_feat_value = feat_value.reshape((feat_value.shape[0], -1))

            #intra-region collection
            representative_feat = reshaped_feat_value.permute(1, 0)[vtopk_table]
            collect_w = torch.matmul(representative_feat,reshaped_feat_value*representative_score).softmax(dim=1)
            collect_rep_feat = representative_feat + torch.matmul(collect_w, reshaped_feat_value.permute(1, 0))
            collect_rep_feat = self.collect_conv(collect_rep_feat.reshape([1,*collect_rep_feat.shape,1]).permute(0,2,1,3)).permute(0,2,1,3).reshape(*collect_rep_feat.shape)

            #inter-region interaction
            inter_region_w = torch.matmul(collect_rep_feat, collect_rep_feat.permute(1,0)*representative_score[vtopk_table]).softmax(dim=1)
            interaction_rep_feat = collect_rep_feat + torch.matmul(inter_region_w, collect_rep_feat)
            interaction_rep_feat = self.interact_conv(interaction_rep_feat.reshape([1,*interaction_rep_feat.shape,1]).permute(0,2,1,3)).permute(0,2,1,3).reshape(*interaction_rep_feat.shape)

            #intra-region distribution
            distribute_w = torch.matmul(reshaped_feat_query.permute(1, 0),interaction_rep_feat.permute(1,0)*representative_score[vtopk_table]).softmax(dim=1)
            distribute_feat = reshaped_feat_query + torch.matmul(distribute_w, interaction_rep_feat).permute(1,0)
            contextual_feat = distribute_feat.reshape(feat_query.shape)
            contextual_feat = self.distribute_conv(contextual_feat.unsqueeze(0)).squeeze(0)
            contextual_feats.append(contextual_feat)

        final_feats = self.final_conv(torch.cat((feats,torch.cat(contextual_feats).reshape(feats.shape)),dim=1))
        return final_feats

class RDNet(ResNet50):
    """Pyramid Scene Parsing Network

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.
    Reference:
        Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang,and Hanqing Lu.
        "Dual Attention Network for Scene Segmentation." *CVPR*, 2019
    """

    def __init__(self, nclass, aux=True, **kwargs):
        super(RDNet, self).__init__(nclass)
        self.head = _DAHead(2048, nclass, aux, **kwargs)
        self.aux = True
        self.__setattr__('exclusive', ['head'])
        

    def forward(self, x):
        size = x.size()[2:]
        feature_map,_ = self.base_forward(x)
        c3,c4 = feature_map[2],feature_map[3]

        # outputs = []
        x = self.head(c3, c4)
        x0 = F.interpolate(x[0], size, mode='bilinear', align_corners=True)
        # outputs.append(x0)

        if self.aux:
            #print('x[1]:{}'.format(x[1].shape))
            # x1 = F.interpolate(x[1], size, mode='bilinear', align_corners=True)
            x2 = F.interpolate(x[2], size, mode='bilinear', align_corners=True)
            # outputs.append(x1)
            # outputs.append(x2)
        return x0, x[1], x2#outputs


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class _DAHead(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )
        self.out1 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, 2, 1)
        )
        # if aux:
        #     self.conv_p3 = nn.Sequential(
        #         nn.Dropout(0.1),
        #         nn.Conv2d(inter_channels, nclass, 1)
        #     )
        #     self.conv_c3 = nn.Sequential(
        #         nn.Dropout(0.1),
        #         nn.Conv2d(inter_channels, nclass, 1)
        #     )
        
        self.head_semantic_logit = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )   #nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)

        self.rcb = RCB()
        self.rib = RIB()

        # self.dsn = nn.Sequential(
        #     nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
        #     norm_layer(512),
        #     nn.Dropout2d(0.1),
        #     )
        # self.dsn_semantic_logit = nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
        # self.dsn_boundary_logit = nn.Conv2d(512, 2, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):

        
        # x_feat = self.dsn(x1)
        # x_semantic_dsn = self.dsn_semantic_logit(x_feat)
        # print(x_semantic_dsn.size())
        # x_boundary_dsn = self.dsn_boundary_logit(x_feat)
        # print(x_boundary_dsn.size())



        feat_p = self.conv_p1(x2)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x2)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        outputs = []
        fusion_out = self.out(feat_fusion)
        fusion_out1 = self.out1(feat_fusion)
        # outputs.append(fusion_out)
        # if self.aux:
        #     p_out = self.conv_p3(feat_p)
        #     c_out = self.conv_c3(feat_c)
            # outputs.append(p_out)
            # outputs.append(c_out)

        region_attention_tables, region_dicision_tables = self.rcb(fusion_out, fusion_out1)
        # print(region_attention_tables.size())
        # print(region_dicision_tables.size())
        # region_attention_tables, region_dicision_tables = self.rcb(x_semantic_dsn, x_boundary_dsn)
        final_feats = self.rib(x2, region_attention_tables, region_dicision_tables)
        x_semantic = self.head_semantic_logit(final_feats + feat_fusion)

        outputs.append(fusion_out)
        outputs.append(torch.cat((feat_p, feat_c), 1))
        outputs.append(x_semantic)
        return tuple(outputs)


if __name__ == '__main__':
    img = torch.randn(2, 3, 480, 480)
    model = get_danet()
    outputs = model(img)
    #print(outputs)

# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

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
            # nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 512, kernel_size=3, padding=1, bias=False),
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

class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)
    

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3,16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
           self.layers.append(non_bottleneck_1d(64, 0.03, 1)) 

        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 2):    #2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        #Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output1 = self.output_conv(output)
            return output, output1

        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64,16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output

#ERFNet
class RERFNet(nn.Module):
    def __init__(self, num_classes, encoder=None):  #use encoder to pass pretrained encoder
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

        self.output_conv = nn.Conv2d(128, 2, 1, stride=1, padding=0, bias=True)


        self.inter_channels = 512

        self.head_semantic_logit = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(self.inter_channels, num_classes, 1)
        )   #nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)

        self.rcb = RCB()
        self.rib = RIB()

        self.pool = nn.AvgPool2d(2, stride=2)

        
        # self.feats_conv = nn.Sequential(
        #     nn.Conv2d(128, 2048, kernel_size=1, padding=0, bias=False),
        #     BatchNorm2d_relu(2048),
        #     # nn.Dropout2d(0.1),
        # )

    def forward(self, input, only_encode=False):
        # if only_encode:
        #     return self.encoder.forward(input, predict=True)
        # else:
        #     output = self.encoder(input)    #predict=False by default
        #     return self.decoder.forward(output)

        # output1 = self.encoder.forward(input, predict=True)
        x, fusion_out1 = self.encoder.forward(input, predict=True)
        fusion_out = self.decoder.forward(x)

        fusion_out2 = self.output_conv(x)
        # outputs.append(fusion_out)
        # if self.aux:
        #     p_out = self.conv_p3(feat_p)
        #     c_out = self.conv_c3(feat_c)
            # outputs.append(p_out)
            # outputs.append(c_out)

        region_attention_tables, region_dicision_tables = self.rcb(self.pool(fusion_out1), self.pool(fusion_out2))
        # print(region_attention_tables.size())
        # print(region_dicision_tables.size())
        # region_attention_tables, region_dicision_tables = self.rcb(x_semantic_dsn, x_boundary_dsn)
        final_feats = self.rib(self.pool(x), region_attention_tables, region_dicision_tables)
        x_semantic = (self.head_semantic_logit(final_feats) + self.pool(fusion_out1)) / 2
        # x_semantic = F.interpolate(x_semantic, fusion_out.size()[2:], mode='bilinear', align_corners=True)
        return x_semantic, fusion_out

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.fanet.resnet import Resnet18,Resnet34,Resnet50,Resnet101,Resnet152


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
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
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


class MyBatchNorm2d(nn.BatchNorm2d):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, num_features, activation='none'):
        super(MyBatchNorm2d, self).__init__(num_features=num_features)
        self.bn = nn.BatchNorm2d(num_features)
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'none':
            self.activation = lambda x:x
        else:
            raise Exception("Accepted activation: ['leaky_relu']")

    def forward(self, x):
        x = self.bn(x)
        return self.activation(x)
 



up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class RFANet(nn.Module):
    def __init__(self,
                 nclass=19,
                 backbone='resnet18',
                 norm_layer=MyBatchNorm2d):
        super(RFANet, self).__init__()
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        self.nclass = nclass
        self.backbone = backbone
        if backbone == 'resnet18':
            self.expansion = 1
            self.resnet = Resnet18(norm_layer=norm_layer)
        elif backbone == 'resnet34':
            self.expansion = 1
            self.resnet = Resnet34(norm_layer=norm_layer)
        elif backbone == 'resnet50':
            self.expansion = 4
            self.resnet = Resnet50(norm_layer=norm_layer)
        elif backbone == 'resnet101':
            self.expansion = 4
            self.resnet = Resnet101(norm_layer=norm_layer)
        elif backbone == 'resnet152':
            self.expansion = 4
            self.resnet = Resnet152(norm_layer=norm_layer)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        self.fam_32 = FastAttModule(512*self.expansion,256,128,norm_layer=norm_layer)
        self.fam_16 = FastAttModule(256*self.expansion,256,128,norm_layer=norm_layer)
        self.fam_8 = FastAttModule(128*self.expansion,256,128,norm_layer=norm_layer)
        self.fam_4 = FastAttModule(64*self.expansion,256,128,norm_layer=norm_layer)

        self.clslayer  = FPNOutput(256, 256, nclass ,norm_layer=norm_layer)
        self.clslayer2  = FPNOutput(256, 256, 2 ,norm_layer=norm_layer)
        self.clslayer3  = FPNOutput(512, 256, nclass ,norm_layer=norm_layer)

        self.rcb = RCB()
        self.rib = RIB()

        # self.head_semantic_logit = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Conv2d(512, nclass, 1)
        # )   #nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)

        self.pool = nn.AvgPool2d(2, stride=2)

        self.feats_conv = nn.Sequential(
            # nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm2d_relu(512),
            # nn.Dropout2d(0.1),
        )

    def forward(self, x, lbl=None):

        _, _, h, w = x.size()

        feat4, feat8, feat16, feat32 = self.resnet(x)
        upfeat_32, smfeat_32, att1 = self.fam_32(feat32,None,True,True)
        upfeat_16, smfeat_16, att2 = self.fam_16(feat16,upfeat_32,True,True)
        upfeat_8, smfeat_8, _ = self.fam_8(feat8,upfeat_16,True,True)
        smfeat_4 = self.fam_4(feat4,upfeat_8,False,True)
        x1 = self._upsample_cat(smfeat_16, smfeat_8)
        x2 = self._upsample_cat(smfeat_16, smfeat_4)
        output1 = self.clslayer2(x2)
        output2 = self.clslayer(x2)

        _,_,H,W = output1.size()

        region_attention_tables, region_dicision_tables = self.rcb(F.interpolate(output2, (H // 2, W // 2), **self._up_kwargs), F.interpolate(output1, (H // 2, W // 2), **self._up_kwargs))
        # print(region_attention_tables.size())
        # print(region_dicision_tables.size())
        # region_attention_tables, region_dicision_tables = self.rcb(x_semantic_dsn, x_boundary_dsn)
        final_feats = self.rib(x1, region_attention_tables, region_dicision_tables)
        # x_semantic = self.head_semantic_logit(self._upsample_add(self.feats_conv(x2), final_feats))
        x_semantic = self.clslayer3(final_feats)


        return output2, att2, x_semantic

    def _upsample_cat(self, x1, x2):
        '''Upsample and concatenate feature maps.
        '''
        _,_,H,W = x2.size()
        x1 = F.interpolate(x1, (H,W), **self._up_kwargs)
        x = torch.cat([x1,x2],dim=1)
        return x

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, (H,W), **self._up_kwargs) + y

    # def get_1x_lr_params_NOscale(self):
    #     """
    #     This generator returns all the parameters of the net except for
    #     the last classification layer. Note that for each batchnorm layer,
    #     requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    #     any batchnorm parameter
    #     """
    #     b = []

    #     b.append(self.resnet)
    #     b.append(self.fam_32)
    #     b.append(self.fam_16)
    #     b.append(self.fam_8)
    #     b.append(self.fam_4)

    #     for i in range(len(b)):
    #         for j in b[i].modules():
    #             jj = 0
    #             for k in j.parameters():
    #                 jj += 1
    #                 if k.requires_grad:
    #                     yield k

    # def get_10x_lr_params(self):
    #     """
    #     This generator returns all the parameters for the last layer of the net,
    #     which does the classification of pixel into classes
    #     """
    #     b = []
    #     b.append(self.clslayer.parameters())
    #     b.append(self.clslayer1.parameters())

    #     for j in range(len(b)):
    #         for i in b[j]:
    #             yield i

    # def optim_parameters(self, args):
    #     return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
    #             {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, norm_layer=None, activation='leaky_relu',*args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.norm_layer = norm_layer
        if self.norm_layer is not None:
            self.bn = norm_layer(out_chan, activation=activation)
        else:
            self.bn =  lambda x:x

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class FPNOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, norm_layer=None, *args, **kwargs):
        super(FPNOutput, self).__init__()
        self.norm_layer = norm_layer
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1, norm_layer=norm_layer)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x



class FastAttModule(nn.Module):
    def __init__(self, in_chan, mid_chn=256, out_chan=128, norm_layer=None, *args, **kwargs):
        super(FastAttModule, self).__init__()
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        mid_chn = int(in_chan/2)        
        self.w_qs = ConvBNReLU(in_chan, 32, ks=1, stride=1, padding=0, norm_layer=norm_layer, activation='none')

        self.w_ks = ConvBNReLU(in_chan, 32, ks=1, stride=1, padding=0, norm_layer=norm_layer, activation='none')

        self.w_vs = ConvBNReLU(in_chan, in_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer)

        self.latlayer3 = ConvBNReLU(in_chan, in_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer)

        self.up = ConvBNReLU(in_chan, mid_chn, ks=1, stride=1, padding=1, norm_layer=norm_layer)
        self.smooth = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1, norm_layer=norm_layer)


    def forward(self, feat, up_fea_in,up_flag, smf_flag):

        query = self.w_qs(feat)
        key   = self.w_ks(feat)
        value = self.w_vs(feat)

        N,C,H,W = feat.size()

        query_ = query.view(N,32,-1).permute(0, 2, 1)
        query = F.normalize(query_, p=2, dim=2, eps=1e-12)  

        key_   = key.view(N,32,-1)
        key   = F.normalize(key_, p=2, dim=1, eps=1e-12)

        value = value.view(N,C,-1).permute(0, 2, 1)

        f = torch.matmul(key, value)
        y = torch.matmul(query, f)
        
        y = y.permute(0, 2, 1).contiguous()

        y = y.view(N, C, H, W)
        W_y = self.latlayer3(y)
        p_feat = W_y + feat
        # att = torch.cat((W_y, feat), dim = 1)

        if up_flag and smf_flag:
            if up_fea_in is not None:
                att = self._upsample_cat(up_fea_in, p_feat)
                p_feat = self._upsample_add(up_fea_in, p_feat)
            else:
                att = p_feat
            up_feat = self.up(p_feat)
            smooth_feat = self.smooth(p_feat)
            return up_feat, smooth_feat, att

        if up_flag and not smf_flag:
            if up_fea_in is not None:
                p_feat = self._upsample_add(up_fea_in, p_feat)
            up_feat = self.up(p_feat)
            return up_feat

        if not up_flag and smf_flag:
            if up_fea_in is not None:
                p_feat = self._upsample_add(up_fea_in, p_feat)
            smooth_feat = self.smooth(p_feat)
            return smooth_feat


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, (H,W), **self._up_kwargs) + y

    def _upsample_cat(self, x1, x2):
        '''Upsample and concatenate feature maps.
        '''
        _,_,H,W = x2.size()
        x1 = F.interpolate(x1, (H,W), **self._up_kwargs)
        x = torch.cat([x1,x2],dim=1)
        return x


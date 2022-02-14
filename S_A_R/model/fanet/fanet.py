import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import Resnet18,Resnet34,Resnet50,Resnet101,Resnet152


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

class FANet(nn.Module):
    def __init__(self,
                 nclass=19,
                 backbone='resnet18',
                 norm_layer=MyBatchNorm2d):
        super(FANet, self).__init__()
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

        self.clslayer1  = FPNOutput(256, 256, nclass,norm_layer=norm_layer)
        self.clslayer  = FPNOutput(256, 256, nclass,norm_layer=norm_layer)

    def forward(self, x, lbl=None):

        _, _, h, w = x.size()

        feat4, feat8, feat16, feat32 = self.resnet(x)
        upfeat_32, smfeat_32, att1 = self.fam_32(feat32,None,True,True)
        upfeat_16, smfeat_16, att2 = self.fam_16(feat16,upfeat_32,True,True)
        upfeat_8, smfeat_8, _ = self.fam_8(feat8,upfeat_16,True,True)
        smfeat_4 = self.fam_4(feat4,upfeat_8,False,True)
        x1 = self._upsample_cat(smfeat_16, smfeat_8)
        x2 = self._upsample_cat(smfeat_16, smfeat_4)
        output1 = self.clslayer1(x1)
        output2 = self.clslayer(x2)
        return output2, att2, output1

    def _upsample_cat(self, x1, x2):
        '''Upsample and concatenate feature maps.
        '''
        _,_,H,W = x2.size()
        x1 = F.interpolate(x1, (H,W), **self._up_kwargs)
        x = torch.cat([x1,x2],dim=1)
        return x

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


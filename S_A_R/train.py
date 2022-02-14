import sys
sys.path.append('~/miniconda3/pkgs')

import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp
import matplotlib.pyplot as plt
import random


from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image, ImageOps
from transform import Relabel, ToLabel, Colorize

# from model.self_attention import *
from model.spectralnorm_discriminator import FCDiscriminator
# from utils.loss import CrossEntropy2d
# from dataset.cityscapes import cityscapesDataSet
# from dataset.cityscapes_rain import cityscapesRainDataSet


from model.fanet.fanet import FANet
from danet import DNet
from rdanet import RDNet
from rfanet import RFANet
from datasets import cityscapes
from iouEval import iouEval, getColorEntry

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 1
DATA_DIRECTORY = 'C:/Users/eggpa/datasets/Cityscapes_extra'
INPUT_SIZE = '1280, 720'#'1024, 512'
DATA_DIRECTORY_TARGET = 'C:/Users/eggpa/datasets/WildPASS2K'
DATA_DIRECTORY_TEST = 'C:/Users/eggpa/dataset/WildPASS'
IGNORE_LABEL = 19
INPUT_SIZE_TARGET = '2048, 400' 
LEARNING_RATE = 2.5e-12
MOMENTUM = 0.9
NUM_CLASSES = 20
NUM_STEPS = 100000
NUM_STEPS_STOP = 100000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 100
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-7
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.001 # 0.0002 
LAMBDA_ADV_TARGET2 = 0.0002 # 0.001
LAMBDA_ADV_TARGET3 = 0.0002 # 0.001
GAN = 'LS'

TARGET = 'snowycs'
SET = 'train'

image_transform = ToPILImage()



class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        #print(outputs.size())
        #print(targets.size())
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)

class MyCoTransform(object):
    def __init__(self, augment=True, height=512, width = 1024, rel = False, target = False):
        self.augment = augment
        self.height = height
        self.width = width
        self.rel = rel
        self.target = target
        pass
    def __call__(self, input):
        # do something to both images
        if (not self.target):
            input =  Resize((self.height, self.width), Image.BILINEAR)(input)
        else:
            input = Resize((self.height, self.width), Image.NEAREST)(input)

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
            
            #Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)

            if (not self.target):
                input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            else:
                input = ImageOps.expand(input, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
        if (not self.target):
            input = ToTensor()(input)
            # print("do not change")
        else:
            input = ToLabel()(input)
            #if self.rel:
            input = Relabel(255, 19)(input)
            # print("do change")
            # if (self.enc):
            #     target = Resize(int(self.height/8), Image.NEAREST)(target)

        return input

def cal_iou(model, testloader, device, args):
    model.eval()
    # if args.num_classes == 19:
    #     iouEvalVal = iouEval(args.num_classes, ignoreIndex=255)
    # else:
    iouEvalVal = iouEval(args.num_classes, ignoreIndex=19)
    interp = nn.Upsample(size=(400, 2048), mode='bilinear', align_corners=True)

    for step, batch in enumerate(testloader):
        images, labels, _, _ = batch
        images = images.to(device)
        labels = labels.unsqueeze(1).to(device)

        with torch.no_grad():
           outputs, _, _ = model(images)
           outputs = interp(outputs)
        iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, labels)

        #filenameSave = filename[0].split("leftImg8bit/")[1] 

        #print(step, filenameSave)

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + ' ' #'\033[0m'
        iou_classes_str.append(iouStr)

    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + ' ' # '\033[0m'

    model.train()
    return iouVal


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-dir-test", type=str, default=DATA_DIRECTORY_TEST,
                        help="Path to the directory containing the test dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target3", type=float, default=LAMBDA_ADV_TARGET3,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                       help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
    return parser.parse_args()


args = get_arguments()


# def loss_calc(pred, label, gpu):
#     """
#     This function returns cross entropy loss for semantic segmentation
#     """
#     # out shape batch_size x channels x h x w -> batch_size x channels x h x w
#     # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
#     label = Variable(label.long()).cuda(gpu)
#     criterion = CrossEntropy2d().cuda(gpu)
   
#     return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    #return -torch.sum(torch.mul(prob, torch.log2(prob + 1e-30))) / (n * h * w * np.log2(c))
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


def main():
    """Create the model and start the training."""

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True
    device = torch.device("cuda" if not args.cpu else "cpu")

    if args.model == 'DeepLab':
        model = DeeplabMulti(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)

        # new_params = model.state_dict().copy()
        # for i in saved_state_dict:
        #     # Scale.layer5.conv2d_list.3.weight
        #     i_parts = i.split('.')
        #     # print i_parts
        #     if not args.num_classes == 19 or not i_parts[1] == 'layer5':
        #         new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        #         # print i_parts
        # model.load_state_dict(new_params)

        own_state = model.state_dict()
        for name, param in saved_state_dict.items():
            if (not name.startswith("layer5.")) and (not name.startswith("layer6.")):
                own_state[name].copy_(param)
            else:
                print(name, " not loaded")
                continue

        # model_dict = model.state_dict()
        # saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
        # model_dict.update(saved_state_dict)
        # ###
        # model.load_state_dict(saved_state_dict)
    elif args.model == 'FANet':
        model = FANet(args.num_classes, backbone = 'resnet34')
        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    if name.startswith("module."):
                        own_state[name.split("module.")[-1]].copy_(param)
                    else:
                        print(name, " not loaded")
                        continue
                else:
                    own_state[name].copy_(param)
            return model

        model = load_my_state_dict(model, torch.load(args.restore_from, map_location=lambda storage, loc: storage))
    elif args.model == 'RFANet':
        model = RFANet(args.num_classes, backbone = 'resnet34')
        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    if name.startswith("module."):
                        own_state[name.split("module.")[-1]].copy_(param)
                    else:
                        print(name, " not loaded")
                        continue
                else:
                    own_state[name].copy_(param)
            return model

        model = load_my_state_dict(model, torch.load(args.restore_from, map_location=lambda storage, loc: storage))
    elif args.model == 'RDANet':
        model = RDNet(args.num_classes)
        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    if name.startswith("module."):
                        own_state[name.split("module.")[-1]].copy_(param)
                    else:
                        print(name, " not loaded")
                        continue
                else:
                    own_state[name].copy_(param)
            return model

        model = load_my_state_dict(model, torch.load(args.restore_from, map_location=lambda storage, loc: storage))
    elif args.model == 'DANet':
        model = DNet(args.num_classes)
        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    if name.startswith("module."):
                        own_state[name.split("module.")[-1]].copy_(param)
                    else:
                        print(name, " not loaded")
                        continue
                else:
                    own_state[name].copy_(param)
            return model

        model = load_my_state_dict(model, torch.load(args.restore_from, map_location=lambda storage, loc: storage))


    model.train()
    model.to(device)

    cudnn.benchmark = True

    # init D
    model_D1 = FCDiscriminator(num_classes=args.num_classes) #Entropy discriminator
    if args.model == 'FANet' or args.model == 'RFANet':
        model_D2 = FCDiscriminator(num_classes=512, last = 2) #Attention discriminator
    elif args.model == 'DANet':
        model_D2 = FCDiscriminator(num_classes=1024, last = 2)
    elif args.model == 'RDANet':
        model_D2 = FCDiscriminator(num_classes=1024, last = 2)

    model_D3 = FCDiscriminator(num_classes=args.num_classes)
    

    model_D1.train()
    model_D1.to(device)

    model_D2.train()
    model_D2.to(device)
    
    model_D3.train()
    model_D3.to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    # trainloader = data.DataLoader(
    #     GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
    #                 crop_size=input_size,
    #                 scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
    #     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # if args.num_classes == 19:
    #     flag = False
    # else:
    flag = True

    print(flag)

    co_transform_1 = MyCoTransform(augment=True, height=input_size[1], width=input_size[0], rel=flag)
    co_transform_2 = MyCoTransform(augment=True, height=input_size_target[1], width=input_size_target[0], rel=flag)
    co_transform_1_t = MyCoTransform(augment=True, height=input_size[1], width=input_size[0], rel=flag, target = True)
    co_transform_2_t = MyCoTransform(augment=True, height=input_size_target[1], width=input_size_target[0], rel=flag, target = True)
    dataset_train = cityscapes(args.data_dir, co_transform_1, co_transform_1_t, subset = 'train')
    dataset_val = cityscapes(args.data_dir_target, co_transform_2, co_transform_2_t, subset = 'train', target = True)
    
    # trainloader = data.DataLoader(
    #     cityscapesDataSet(args.data_dir, max_iters=args.num_steps * args.iter_size * args.batch_size,
    #                 crop_size=input_size,
    #                 scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
    #     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    # targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
    #                                                 max_iters=args.num_steps * args.iter_size * args.batch_size,
    #                                                 crop_size=input_size_target,
    #                                                 scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
    #                                                 set=args.set),
    #                             batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    #                             pin_memory=True)

    # targetloader = data.DataLoader(
    #     cityscapesDataSet(args.data_dir_target, max_iters=args.num_steps * args.iter_size * args.batch_size,
    #                 crop_size=input_size_target,
    #                 scale=False, mirror=args.random_mirror, mean=IMG_MEAN, target = True),
    #     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    #     pin_memory=True)

    targetloader = data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    targetloader_iter = enumerate(targetloader)

    # testloader = data.DataLoader(cityscapesDataSet(args.data_dir_test, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set='val', target = False),
    #                                 batch_size=1, shuffle=False, pin_memory=True)

    input_transform_cityscapes = Compose([
    Resize((400, 2048), Image.BILINEAR),
    ToTensor(),
    ])
    #if (flag):
    target_transform_cityscapes = Compose([
        Resize((400, 2048), Image.NEAREST),
        ToLabel(),
        Relabel(255, 19),   #ignore label to 19
    ])
    # else:
    #     target_transform_cityscapes = Compose([
    #         Resize((512, 1024), Image.NEAREST),
    #         ToLabel(),
    #     ])

    testloader = data.DataLoader(cityscapes(args.data_dir_test, input_transform_cityscapes, target_transform_cityscapes, subset='val', target=False), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    
    optimizer_D3 = optim.Adam(model_D3.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D3.zero_grad()

    if args.gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif args.gan == 'LS':
        bce_loss = torch.nn.MSELoss()

    weight = torch.ones(NUM_CLASSES)
    weight[0] = 2.8149201869965	
    weight[1] = 6.9850029945374	
    weight[2] = 3.7890393733978	
    weight[3] = 9.9428062438965	
    weight[4] = 9.7702074050903	
    weight[5] = 9.5110931396484	
    weight[6] = 10.311357498169	
    weight[7] = 10.026463508606	
    weight[8] = 4.6323022842407	
    weight[9] = 9.5608062744141	
    weight[10] = 7.8698215484619	
    weight[11] = 9.5168733596802	
    weight[12] = 10.373730659485	
    weight[13] = 6.6616044044495	
    weight[14] = 10.260489463806	
    weight[15] = 10.287888526917	
    weight[16] = 10.289801597595	
    weight[17] = 10.405355453491	
    weight[18] = 10.138095855713	
    weight[19] = 0
    weight = weight.to(device)
    seg_loss = CrossEntropyLoss2d(weight)

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1

    b_best = cal_iou(model, testloader, device, args)
    best = 0
    print(f'iou before adaptation: {b_best * 100:.3f}%')

    for i_iter in range(0, args.num_steps):

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        loss_adv_target_value3 = 0
        loss_D_value3 = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        optimizer_D3.zero_grad()
        adjust_learning_rate_D(optimizer_D1, i_iter)
        adjust_learning_rate_D(optimizer_D2, i_iter)
        adjust_learning_rate_D(optimizer_D3, i_iter)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False

            for param in model_D2.parameters():
                param.requires_grad = False

            # train with source

            # _, batch = trainloader_iter.__next__()
            # images, labels = batch
            # labels1 = labels
            # labels1[labels1!=0]=1
            # labels1 = labels1.cuda(args.gpu)
            # images = Variable(images).cuda(args.gpu)

            try:
                _, batch = trainloader_iter.__next__()
            except StopIteration:
                trainloader_iter = enumerate(trainloader)
                _, batch = trainloader_iter.__next__()

            images, labels, _, _ = batch
            images = images.to(device)
            labels = labels.to(device)

            out, out_attn, pred2 = model(images)
            
            out = interp(out)
            pred2 = interp(pred2)


            # loss_seg2 = loss_calc(out, labels1, args.gpu)
            loss_seg1 = seg_loss(pred2, labels)
            loss_seg2 = seg_loss(out, labels)
            loss = loss_seg2 + args.lambda_seg * loss_seg1

			
			# proper normalization
            loss = loss / args.iter_size
            loss.backward()
            # loss_seg_value2 += loss_seg2.data.cpu().numpy() / args.iter_size
            loss_seg_value1 += loss_seg1.item() / args.iter_size
            loss_seg_value2 += loss_seg2.item() / args.iter_size

            # torch.cuda.empty_cache()
        

            # train with target

            # _, batch = targetloader_iter.__next__()
            # images, labels = batch
            # images = Variable(images).cuda(args.gpu)            
            try:
                _, batch = targetloader_iter.__next__()
            except StopIteration:
                targetloader_iter = enumerate(targetloader)
                _, batch = targetloader_iter.__next__()
            images, _, _ = batch
            images = images.to(device)

            out_target, out_attn_target, pred_target2 = model(images)
            
            out_target = interp_target(out_target)
            pred_target2 = interp_target(pred_target2)

            # D_out1 = model_D1(prob_2_entropy(F.softmax(out_target, dim=1)))
            D_out1 = model_D1(F.softmax(out_target, dim=1))
            D_out2 = model_D2(F.softmax(out_attn_target, dim=1))
            D_out3 = model_D3(F.softmax(pred_target2, dim=1))

            loss_adv_target1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))

            loss_adv_target2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))

            loss_adv_target3 = bce_loss(D_out3, torch.FloatTensor(D_out3.data.size()).fill_(source_label).to(device))

            loss = args.lambda_adv_target1 * loss_adv_target1 + args.lambda_adv_target2 * loss_adv_target2 + args.lambda_adv_target3 * loss_adv_target3
            loss = loss / args.iter_size
            loss.backward()
            # loss_adv_target_value1 += loss_adv_target1.data.cpu().numpy() / args.iter_size
            # loss_adv_target_value2 += loss_adv_target2.data.cpu().numpy() / args.iter_size
            loss_adv_target_value1 += loss_adv_target1.item() / args.iter_size
            loss_adv_target_value2 += loss_adv_target2.item() / args.iter_size
            loss_adv_target_value3 += loss_adv_target3.item() / args.iter_size

            # train D

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            for param in model_D2.parameters():
                param.requires_grad = True

            for param in model_D3.parameters():
                param.requires_grad = True

            # train with source
            out = out.detach()
            out_attn = out_attn.detach()
            pred2 = pred2.detach()

            # D_out1 = model_D1(prob_2_entropy(F.softmax(out, dim=1)))
            D_out1 = model_D1(F.softmax(out, dim=1))
            D_out2 = model_D2(F.softmax(out_attn, dim=1))
            D_out3 = model_D3(F.softmax(pred2, dim=1))

            loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))

            loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))

            loss_D3 = bce_loss(D_out3, torch.FloatTensor(D_out3.data.size()).fill_(source_label).to(device))

            loss_D1 = loss_D1 / args.iter_size / 3
            loss_D2 = loss_D2 / args.iter_size / 3
            loss_D3 = loss_D3 / args.iter_size / 3

            loss_D1.backward()
            loss_D2.backward()
            loss_D3.backward()

            loss_D_value1 += loss_D1.item()
            loss_D_value2 += loss_D2.item()
            loss_D_value3 += loss_D3.item()

            # train with target
            out_target = out_target.detach()
            out_attn_target = out_attn_target.detach()
            pred_target2 = pred_target2.detach()

            D_out1 = model_D1(prob_2_entropy(F.softmax(out_target, dim=1)))
            D_out2 = model_D2(F.softmax(out_attn_target, dim=1))
            D_out3 = model_D3(F.softmax(pred_target2, dim=1))


            loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(target_label).to(device))

            loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(target_label).to(device))

            loss_D3 = bce_loss(D_out3, torch.FloatTensor(D_out3.data.size()).fill_(target_label).to(device))

            loss_D1 = loss_D1 / args.iter_size / 3
            loss_D2 = loss_D2 / args.iter_size / 3
            loss_D3 = loss_D3 / args.iter_size / 3

            loss_D1.backward()
            loss_D2.backward()
            loss_D3.backward()

            loss_D_value1 += loss_D1.item()
            loss_D_value2 += loss_D2.item()
            loss_D_value3 += loss_D3.item()

        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()
        optimizer_D3.step()

        # torch.cuda.empty_cache()
        
        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} loss_adv1 = {4:.3f}, loss_adv2 = {5:.3f} loss_adv3 = {10:.3f} loss_D1 = {6:.3f} loss_D2 = {7:.3f} loss_D3 = {11:.3f} cur_best = {8:.3f}% b_best = {9:.3f}%'.format(
            i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, loss_adv_target_value1, loss_adv_target_value2, loss_D_value1, loss_D_value2, best * 100, b_best * 100, loss_adv_target3, loss_D_value3))

        # if i_iter >= args.num_steps_stop - 1:
        #     print('save model ...')
        #     torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
        #     torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D1.pth'))
        #     torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D2.pth'))
        #     break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            iou = cal_iou(model, testloader, device, args)
            print(f'iou after {i_iter} iterantions :{iou * 100:.3f}%')
            if (iou > best):
                print(f'saving best model ...')
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'best_adpt.pth'))
                best = iou
                #torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D1.pth'))
                #torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D2.pth'))


if __name__ == '__main__':
    main()



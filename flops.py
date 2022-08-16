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
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
from thop import profile

# from model.deeplab_multi import DeeplabMulti
from model.deeplab_multi import DeeplabMulti_output, DeeplabMulti_CLAN
from model.discriminator import FCDiscriminator_output

from model.deeplab_cca import DeeplabMulti, CCA_Classifier_V1
from model.discriminator import FCDiscriminator_CCA1
from utils.loss import CrossEntropy2d
# from dataset.gta5_dataset import GTA5DataSet
# from dataset.cityscapes_dataset import cityscapesDataSet

from dataset.isprs_dataset import ISPRSDataset, ISPRSDataset_val
import albumentations as albu
from metrics import StreamSegMetrics

# IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
# DATA_DIRECTORY = './data/GTA5'
# DATA_LIST_PATH = './dataset/gta5_list/train.txt'

IMAGE_DIRECTORY = "./dataset/Potsdam/train_img_960"
LABEL_DIRECTORY = "./dataset/Potsdam/train_lab_960"

IMAGE_DIRECTORY_TARGET = "./dataset/Vaihingen/train_img_512"
LABEL_DIRECTORY_TARGET = "./dataset/Vaihingen/train_lab_512"
# IMAGE_DIRECTORY_TARGET = "./dataset/Vaihingen/val_img"
# LABEL_DIRECTORY_TARGET = "./dataset/Vaihingen/val_lab"
VAL_IMAGE_DIRECTORY_TARGET = "./dataset/Vaihingen/val_img_512"
VAL_LABEL_DIRECTORY_TARGET = "./dataset/Vaihingen/val_lab_512"

IGNORE_LABEL = 255
INPUT_SIZE = '960,960'
# DATA_DIRECTORY_TARGET = './data/Cityscapes/data'
# DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
INPUT_SIZE_TARGET = '512,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 6
NUM_STEPS = 250000
NUM_STEPS_STOP = 150000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './checkpoints_pot2vai_test/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
# LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET = 0.001

TARGET = 'cityscapes'
SET = 'train'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
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

    parser.add_argument("--image-dir", type=str, default=IMAGE_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--label-dir", type=str, default=LABEL_DIRECTORY,
                        help="Path to the directory containing the source dataset.")

    parser.add_argument("--target-image-dir", type=str, default=IMAGE_DIRECTORY_TARGET,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--target-label-dir", type=str, default=LABEL_DIRECTORY_TARGET,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--val-target-image-dir", type=str, default=VAL_IMAGE_DIRECTORY_TARGET,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--val-target-label-dir", type=str, default=VAL_LABEL_DIRECTORY_TARGET,
                        help="Path to the directory containing the source dataset.")

    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
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
    parser.add_argument("--lambda-adv-target", type=float, default=LAMBDA_ADV_TARGET,
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
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)


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

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=45, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=384, min_width=384, always_apply=True, border_mode=0),
        albu.RandomCrop(height=384, width=384, p=0.5),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=360, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=384, min_width=384, always_apply=True, border_mode=0),
        albu.RandomCrop(height=384, width=384, p=0.2),

    ]
    return albu.Compose(test_transform)

def validate(args, model,classifier, model_D, validloader, metrics, interp):
    """Do validation and return specified samples"""
    metrics.reset()

    with torch.no_grad():
        for i, data in enumerate(validloader, start=0):
            # images, labels = validloader_iter.next()
            images, labels = data[0],data[1]
            images = Variable(images).cuda(args.gpu)

            feature = model(images)  # [1,6,33,33][1, 6, 65, 65]
            feature_interp = nn.Upsample(size=(feature.size()[2], feature.size()[3]), mode='bilinear')
            # 121,121
            global_out, pred_D, class_out = model_D(feature)
            D_out = feature_interp(1-F.sigmoid(class_out))  # max:0.0889 min:-0.0637
            pred1, pred = classifier(feature, D_out)
            # proper normalization
            pred1 = interp(pred1)
            outputs = interp(pred)

            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

        score = metrics.get_results()
    return score

class Ours_Model(nn.Module):
    def __init__(self, num_classes):
        super(Ours_Model, self).__init__()

        self.model = DeeplabMulti(num_classes=num_classes)
        self.model_D = FCDiscriminator_CCA1(num_classes=num_classes)
        self.classifier = CCA_Classifier_V1(num_classes=num_classes)
    def forward(self, x):
        feature = self.model(x)
        global_out, pred_D, class_out = self.model_D(feature)
        feature_interp = nn.Upsample(size=(feature.size()[2], feature.size()[3]), mode='bilinear')
        D_out = feature_interp(F.sigmoid(class_out))
        pred1, pred = self.classifier(feature, D_out)
        # pred = interp(pred)
        return pred
class Output_Model(nn.Module):
    def __init__(self, num_classes):
        super(Output_Model, self).__init__()

        self.model = DeeplabMulti_output(num_classes=args.num_classes)
        self.model_D1 = FCDiscriminator_output(num_classes=args.num_classes)
        self.model_D2 = FCDiscriminator_output(num_classes=args.num_classes)

    def forward(self, x):
        pred1, pred2 = self.model(x)
        w, h = map(int, args.input_size.split(','))
        input_size = (w, h)
        interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
        pred1 = interp(pred1)
        pred2 = interp(pred2)
        D_out1 = self.model_D1(F.softmax(pred1))
        D_out2 = self.model_D2(F.softmax(pred2))
        return D_out1, D_out2
class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf = 64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        return x
class CLAN_Model(nn.Module):
    def __init__(self, num_classes):
        super(CLAN_Model, self).__init__()

        self.model = DeeplabMulti_CLAN(num_classes=args.num_classes)
        self.model_D = FCDiscriminator(num_classes=args.num_classes)
        # self.classifier = CCA_Classifier_V1(num_classes=num_classes)
    def forward(self, x):
        pred_source1, pred_source2 = self.model(x)

        pred_source1 = pred_source1
        pred_source2 = pred_source2
        D_out_s = self.model_D(F.softmax(pred_source1 + pred_source2, dim=1))

        return D_out_s

def main():
    """Create the model and start the training."""

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)#(960, 960)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)#(512, 512)

    cudnn.enabled = True
    gpu = args.gpu

    # Create network
    #ours
    # new_model = Our_Model(num_classes=args.num_classes)
    #AdaptSegNet
    # new_model = Output_Model(num_classes=args.num_classes)
    #deeplab v2
    # new_model = DeeplabMulti(num_classes=args.num_classes)
    #CLAN
    new_model = CLAN_Model(num_classes=args.num_classes)
    new_model.cuda(args.gpu)

    # dataset
    train_dataset = ISPRSDataset(
        args.image_dir,
        args.label_dir,
        max_iters=args.num_steps * args.iter_size * args.batch_size,
    )
    trainloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = iter(trainloader)

    target_dataset = ISPRSDataset(
        args.target_image_dir,
        args.target_label_dir,

        max_iters=args.num_steps * args.iter_size * args.batch_size,
    )
    targetloader = data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True)
    targetloader_iter = iter(targetloader)

    val_target_dataset = ISPRSDataset_val(
        args.val_target_image_dir,
        args.val_target_label_dir
    )
    val_targetloader = data.DataLoader(val_target_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=True)

    # implement model.optim_parameters(args) to handle different models' lr setting

    for i_iter in range(args.num_steps):
        for sub_i in range(args.iter_size):

            # train with source
            images, labels, path = trainloader_iter.next()
            images = Variable(images).cuda(args.gpu)
            flops, params = profile(new_model, inputs=(images,))
            print('flops:',flops)
            print('params', params)

if __name__ == '__main__':
    main()

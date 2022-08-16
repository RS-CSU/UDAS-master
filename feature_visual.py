import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path as osp
import seaborn as sns

from model.deeplab_cca import DeeplabMulti
from model.discriminator import FCDiscriminator_CCA1, FCDiscriminator_CA
from utils.loss import CrossEntropy2d

from dataset.isprs_dataset import ISPRSDataset, ISPRSDataset_val
import albumentations as albu
from metrics import StreamSegMetrics
import glob
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn import manifold
from PIL import Image
import torch.nn.functional as F
import cv2
from matplotlib import cm
#matplotlib inline


# IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4

IGNORE_LABEL = 255
INPUT_SIZE = '512,512'
# # DATA_DIRECTORY_TARGET = './data/Cityscapes/data'
# # DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
INPUT_SIZE_TARGET = '512,512'
# LEARNING_RATE = 2.5e-4
# MOMENTUM = 0.9
NUM_CLASSES = 6
# NUM_STEPS = 250000
# NUM_STEPS_STOP = 250000  # early stopping
# POWER = 0.9
# RANDOM_SEED = 1234
# RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
# SAVE_NUM_IMAGES = 2
# SAVE_PRED_EVERY = 5000
# SNAPSHOT_DIR = './checkpoints_pot2vai_CA/'
# WEIGHT_DECAY = 0.0005
#
# LEARNING_RATE_D = 1e-4
# LAMBDA_SEG = 0.1
# # LAMBDA_ADV_TARGET1 = 0.0002
# LAMBDA_ADV_TARGET = 0.001
#
# TARGET = 'cityscapes'
# SET = 'train'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    # parser.add_argument("--target", type=str, default=TARGET,
    #                     help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")

    # parser.add_argument("--image-dir", type=str, default=IMAGE_DIRECTORY,
    #                     help="Path to the directory containing the source dataset.")
    # parser.add_argument("--label-dir", type=str, default=LABEL_DIRECTORY,
    #                     help="Path to the directory containing the source dataset.")
    #
    # parser.add_argument("--target-image-dir", type=str, default=IMAGE_DIRECTORY_TARGET,
    #                     help="Path to the directory containing the source dataset.")
    # parser.add_argument("--target-label-dir", type=str, default=LABEL_DIRECTORY_TARGET,
    #                     help="Path to the directory containing the source dataset.")
    # parser.add_argument("--val-target-image-dir", type=str, default=VAL_IMAGE_DIRECTORY_TARGET,
    #                     help="Path to the directory containing the source dataset.")
    # parser.add_argument("--val-target-label-dir", type=str, default=VAL_LABEL_DIRECTORY_TARGET,
    #                     help="Path to the directory containing the source dataset.")

    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    # parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
    #                     help="Comma-separated string with height and width of target images.")
    # parser.add_argument("--is-training", action="store_true",
    #                     help="Whether to updates the running means and variances during the training.")
    # parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
    #                     help="Base learning rate for training with polynomial decay.")
    # parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
    #                     help="Base learning rate for discriminator.")
    # parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
    #                     help="lambda_seg.")
    # parser.add_argument("--lambda-adv-target", type=float, default=LAMBDA_ADV_TARGET,
    #                     help="lambda_adv for adversarial training.")
    # parser.add_argument("--momentum", type=float, default=MOMENTUM,
    #                     help="Momentum component of the optimiser.")
    # parser.add_argument("--not-restore-last", action="store_true",
    #                     help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    # parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
    #                     help="Number of training steps.")
    # parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
    #                     help="Number of training steps for early stopping.")
    # parser.add_argument("--power", type=float, default=POWER,
    #                     help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    # parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
    #                     help="Random seed to have reproducible results.")
    # parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
    #                     help="Where restore model parameters from.")
    # parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
    #                     help="How many images to save.")
    # parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
    #                     help="Save summaries and checkpoint every often.")
    # parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
    #                     help="Where to save snapshots of the model.")
    # parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
    #                     help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    # parser.add_argument("--set", type=str, default=SET,
    #                     help="choose adaptation set.")
    return parser.parse_args()


args = get_arguments()

normMean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
normStd = np.array((0.229, 0.224, 0.225), dtype=np.float32)

def standardization(image):
    image = ((image / 255) - normMean) / normStd
    return image

def get_images(image_dir, label_dir):
    # read data
    image = Image.open(image_dir).convert('RGB')
    mask = Image.open(label_dir).convert('L')

    image = np.asarray(image, np.float32)
    mask = np.asarray(mask, np.float32)
    image = standardization(image)

    image = image.transpose((2, 0, 1))
    # print(len(self.img_ids))

    return torch.from_numpy(image.copy()).unsqueeze(0), torch.from_numpy(mask.copy()).unsqueeze(0)

def draw_attention():
    image_dir_s = './dataset/Vaihingen/visual_img/'
    label_dir_s = './dataset/Vaihingen/visual_lab/1.png'
    image_dir_t = './dataset/Vaihingen/visual_img/211.png'
    label_dir_t = './dataset/Vaihingen/visual_lab/7.png'

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)  # (960, 960)

    model = DeeplabMulti(num_classes=args.num_classes)
    # save_path = './checkpoints_potsdam/potsdam_best.pth'
    save_path = './checkpoints_pot2vai_class_wiseD_CCA_V3_with3/pot2vai_best.pth'
    model.load_state_dict(torch.load(save_path))
    model.cuda(args.gpu)
    cudnn.benchmark = True
    model.eval()

    # init D
    model_D = FCDiscriminator_CCA1(num_classes=args.num_classes)
    save_path = './checkpoints_pot2vai_class_wiseD_CCA_V3_with3/pot2vai_best_D.pth'
    model_D.load_state_dict(torch.load(save_path))
    model_D.cuda(args.gpu)
    model_D.eval()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')

    fls_list = glob.glob(image_dir_s+'*.png')
    classes = ['Impervious Surfaces','Buildings','Low Vegetation','Tree','Car','Clutter']

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 14, }
    with torch.no_grad():
        for fls in fls_list:
            # source
            images_s, labels_s = get_images(fls, fls.replace('visual_img','visual_lab'))
            img = Image.open(fls)
            images_s = Variable(images_s).cuda(args.gpu)
            # feature_s = model(images_s)
            feature_s = model(images_s)
            global_out, pred_D, class_out = model_D(feature_s)
            out_interp = nn.Upsample(size=(global_out.size()[2], global_out.size()[3]), mode='bilinear')
            # D_out = global_out * out_interp(class_out)
            D_out = torch.mul(global_out, out_interp(class_out))

            attention_class_out = interp(class_out).squeeze(0).cpu().numpy()
            attention_global_out = interp(global_out).squeeze(0).cpu().numpy()
            attention_D = interp(D_out).squeeze(0).cpu().numpy()
            #6,960,960

            plt.figure(1)
            for i in range(attention_class_out.shape[0]):
                ax = plt.subplot(2, 3,i+1)
                plt.title(classes[i],font1)
                plt.sca(ax)
                # plt.legend()
                plt.xticks([])
                plt.yticks([])
                plt.imshow(img)
                plt.imshow(attention_class_out[i,:,:], alpha=0.5, label = 'value')

            plt.savefig(fls.replace('visual_img','visual_result'))

if __name__ == '__main__':
    draw_attention()
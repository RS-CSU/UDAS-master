import argparse
import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os

# from model.deeplab_multi import DeeplabMulti
from model.deeplab_cca import DeeplabMulti, CCA_Classifier
from model.discriminator import FCDiscriminator_CCA
from utils.loss import CrossEntropy2d
from dataset.isprs_dataset import ISPRSDataset, ISPRSDataset_val
from metrics import StreamSegMetrics

import util.util as util
from util.visualizer import Visualizer
from util import html
from collections import OrderedDict
import torch.nn.functional as F

# IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4

IMAGE_DIRECTORY = "./dataset/Potsdam/train_img_960"
LABEL_DIRECTORY = "./dataset/Potsdam/train_lab_960"

IMAGE_DIRECTORY_TARGET = "./dataset/Vaihingen/train_img_512"
LABEL_DIRECTORY_TARGET = "./dataset/Vaihingen/train_lab_512"
# IMAGE_DIRECTORY_TARGET = "./dataset/Vaihingen/val_img"
# LABEL_DIRECTORY_TARGET = "./dataset/Vaihingen/val_lab"
VAL_IMAGE_DIRECTORY_TARGET = "./dataset/Vaihingen/visual_img"
VAL_LABEL_DIRECTORY_TARGET = "./dataset/Vaihingen/visual_lab"

IGNORE_LABEL = 255
INPUT_SIZE = '960,960'
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
SNAPSHOT_DIR = './checkpoints_pot2vai_output/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001

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
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
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
    parser.add_argument("--gpu", type=int, default=1,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--ckpt", type=str, default='suibian',
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


def main():
    """Create the model and start the training."""

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True
    gpu = args.gpu

    # Create network
    if args.model == 'DeepLab':
        model = DeeplabMulti(num_classes=args.num_classes)
        save_path = './checkpoints_pot2vai/pot2vai_best.pth'
        model.load_state_dict(torch.load(save_path))
        print('load success')
    model.cuda(args.gpu)

    model_D = FCDiscriminator_CCA(num_classes=args.num_classes)
    save_path_D = './checkpoints_pot2vai/pot2vai_best_D.pth'
    model_D.load_state_dict(torch.load(save_path_D))
    model_D.cuda(args.gpu)

    classifier = CCA_Classifier(num_classes=args.num_classes)
    save_path_classifier = './checkpoints_pot2vai/pot2vai_best_classifier.pth'
    classifier.load_state_dict(torch.load(save_path_classifier))
    classifier.cuda(args.gpu)

    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    # dataset
    val_target_dataset = ISPRSDataset_val(
        args.val_target_image_dir,
        args.val_target_label_dir,
    )
    val_targetloader = data.DataLoader(val_target_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=True)

    # implement model.optim_parameters(args) to handle different models' lr setting

    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear')

    visualizer = Visualizer(args)
    # create website
    web_dir = os.path.join('./results')
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % ('domain_adaptation', 'test', 'latest'))

    # Labels for Adversarial Training
    metrics = StreamSegMetrics(args.num_classes)
    metrics.reset()

    with torch.no_grad():
        for i, val_data in enumerate(val_targetloader, start=0):
            # images, labels = validloader_iter.next()
            images, labels, path = val_data[0], val_data[1],  val_data[2]
            images = Variable(images).cuda(args.gpu)

            feature = model(images)
            feature_interp = nn.Upsample(size=(feature.size()[2], feature.size()[3]), mode='bilinear')
            # 121,121
            global_out, pred_D, class_out = model_D(feature)
            D_out = feature_interp(1-F.sigmoid(class_out))  # max:0.0889 min:-0.0637
            pred1, pred = classifier(feature, D_out)
            outputs = interp_target(pred)

            new_output = torch.squeeze(outputs, 0)
            new_image = torch.squeeze(images, 0)

            visuals = OrderedDict([('input_label', util.tensor2label(labels, args.num_classes)),
                                   ('synthesized_label', util.tensor2label(new_output, args.num_classes)),
                                   ('real_image', util.tensor2im(new_image))])
            img_path = path[0]
            print('process image... %s' % img_path)
            visualizer.save_images(webpage, visuals, img_path)

            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
        score = metrics.get_results()
        print(score)

if __name__ == '__main__':
    main()

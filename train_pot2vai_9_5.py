import argparse
import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp

from model.deeplab_cca import DeeplabMulti, CCA_Classifier
from model.discriminator import FCDiscriminator_CCA
from utils.loss import CrossEntropy2d

from dataset.isprs_dataset import ISPRSDataset, ISPRSDataset_val
import albumentations as albu
from metrics import StreamSegMetrics

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4

IMAGE_DIRECTORY = "./dataset/Potsdam/train_img_960"
LABEL_DIRECTORY = "./dataset/Potsdam/train_lab_960"

IMAGE_DIRECTORY_TARGET = "./dataset/Vaihingen/train_img_512"
LABEL_DIRECTORY_TARGET = "./dataset/Vaihingen/train_lab_512"
VAL_IMAGE_DIRECTORY_TARGET = "./dataset/Vaihingen/val_img_512"
VAL_LABEL_DIRECTORY_TARGET = "./dataset/Vaihingen/val_lab_512"

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
SNAPSHOT_DIR = './checkpoints_pot2vai/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
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
            images, labels = data[0],data[1]
            images = Variable(images).cuda(args.gpu)

            feature = model(images)  # [1,6,33,33][1, 6, 65, 65]
            feature_interp = nn.Upsample(size=(feature.size()[2], feature.size()[3]), mode='bilinear')
            # 121,121
            global_out, pred_D, class_out = model_D(feature)
            D_out = feature_interp(1-F.sigmoid(class_out))  # max:0.0889 min:-0.0637
            pred1, pred = classifier(feature, D_out)
            # proper normalization
            outputs = interp(pred)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

        score = metrics.get_results()
    return score

def main():
    """Create the model and start the training."""

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)#(960, 960)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)#(512, 512)

    cudnn.enabled = True

    # Create network
    if args.model == 'DeepLab':
        model = DeeplabMulti(num_classes=args.num_classes)
        #加载预训练模型：在源域数据集上训练的模型
        save_path = './checkpoints_potsdam/potsdam_best.pth'
        model.load_state_dict(torch.load(save_path,map_location='cuda:0'))
        print('load success')
    #特征提取器
    model.train()
    model.cuda(args.gpu)
    cudnn.benchmark = True

    # init D：判别器
    model_D = FCDiscriminator_CCA(num_classes=args.num_classes)
    model_D.train()
    model_D.cuda(args.gpu)
    #分类器
    classifier = CCA_Classifier(num_classes=args.num_classes)
    classifier.train()
    classifier.cuda(args.gpu)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

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
    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_classifier = optim.SGD(classifier.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_classifier.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear')

    # labels for adversarial training
    source_label = 0
    target_label = 1

    #评价指标
    metrics = StreamSegMetrics(args.num_classes)
    #实时记录验证集的最高得分
    acc_path = os.path.join(args.snapshot_dir, 'acc.txt')
    # 实时记录每一次验证的得分
    acc_iter_path = os.path.join(args.snapshot_dir, 'acc_iter.txt')
    best_score = 0.0
    loss_seg_value = 0
    loss_adv_target_value = 0
    loss_D_value = 0

    for i_iter in range(args.num_steps):

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_classifier.zero_grad()
        adjust_learning_rate(optimizer_classifier, i_iter)

        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        for sub_i in range(args.iter_size):

            # train G
            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False

            # train with source
            images, labels, path = trainloader_iter.next()
            images = Variable(images).cuda(args.gpu)

            feature = model(images)#[1,6,33,33][1, 6, 65, 65]
            feature_interp = nn.Upsample(size=(feature.size()[2], feature.size()[3]), mode='bilinear')
            # 121,121
            global_out, pred_D, class_out = model_D(feature)
            D_out = feature_interp(F.sigmoid(class_out))  # max:0.0889 min:-0.0637
            pred1, pred = classifier(feature, D_out)
            # proper normalization
            pred1 = interp(pred1)
            pred = interp(pred)
            loss_seg = loss_calc(pred, labels, args.gpu)
            loss = loss_seg

            # proper normalization
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_value += loss_seg.data.cpu().numpy() / args.iter_size

            # train with target
            images, _, _ = targetloader_iter.next()
            images = Variable(images).cuda(args.gpu)

            feature_target = model(images)#[1,6,33,33]
            global_out, pred_D, class_out = model_D(feature_target)#
            loss_adv_target_global = bce_loss(global_out,
                                       Variable(torch.FloatTensor(global_out.data.size()).fill_(source_label)).cuda(
                                           args.gpu))

            loss_adv_target_class = bce_loss(class_out,
                                        Variable(torch.FloatTensor(class_out.data.size()).fill_(source_label)).cuda(
                                            args.gpu))

            loss = args.lambda_adv_target * (loss_adv_target_global + loss_adv_target_class)
            loss = loss / args.iter_size
            loss.backward()
            loss_adv_target_value += (loss_adv_target_global.data.cpu().numpy()
                                      +loss_adv_target_class.data.cpu().numpy()) / args.iter_size

            # train D
            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            # train with source
            feature = feature.detach()
            global_out, pred_D, class_out = model_D(feature)

            loss_D_global = bce_loss(global_out,
                               Variable(torch.FloatTensor(global_out.data.size()).fill_(source_label)).cuda(args.gpu))
            pred_D = interp(pred_D)
            loss_seg = loss_calc(pred_D, labels, args.gpu)
            loss_D_class = bce_loss(class_out,
                                     Variable(torch.FloatTensor(class_out.data.size()).fill_(source_label)).cuda(
                                         args.gpu))
            loss_D = (loss_D_global + loss_D_class) / args.iter_size / 2
            loss = loss_seg + loss_D
            loss.backward()
            loss_D_value += loss.data.cpu().numpy()

            # train with target
            feature_target = feature_target.detach()
            global_out, pred_D, class_out = model_D(feature_target)
            loss_D_global = bce_loss(global_out,
                               Variable(torch.FloatTensor(global_out.data.size()).fill_(target_label)).cuda(args.gpu))
            loss_D_class = bce_loss(class_out,
                                     Variable(torch.FloatTensor(class_out.data.size()).fill_(target_label)).cuda(
                                         args.gpu))

            loss_D = (loss_D_global + loss_D_class) / args.iter_size / 2
            loss_D.backward()
            loss_D_value += loss_D.data.cpu().numpy()
        optimizer.step()
        optimizer_classifier.step()
        optimizer_D.step()
        if (i_iter) % 10 == 0:
            # loss_seg_value = loss_seg_value / 10
            print('exp = {}'.format(args.snapshot_dir))
            print(
                'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv = {3:.3f} loss_D = {4:.3f}'.format(
                    i_iter, args.num_steps, loss_seg_value/ 10, loss_adv_target_value/ 10, loss_D_value/ 10))
            loss_seg_value = 0.0
            loss_adv_target_value = 0.0
            loss_D_value = 0.0

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'pot2vai_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'pot2vai_' + str(args.num_steps_stop) + '_D.pth'))
            torch.save(classifier.state_dict(),
                       osp.join(args.snapshot_dir, 'pot2vai_' + str(args.num_steps_stop) + '_classifier.pth'))
            break

        #每迭代300次，验证一次
        if i_iter % 300 == 0:
            print("validation...")
            val_score = validate(
                args, model=model, classifier = classifier, model_D = model_D, validloader=val_targetloader, metrics=metrics, interp=interp_target)
            print(metrics.to_str(val_score))

            fh = open(acc_iter_path, 'a')
            fh.write('iter:' + str(i_iter))
            fh.write(metrics.to_str(val_score))
            fh.close()

            if val_score['Overall Acc'] > best_score:  # save best model
                best_score = val_score['Overall Acc']
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'pot2vai_' + str(best_score) + '.pth'))
                torch.save(model_D.state_dict(),
                           osp.join(args.snapshot_dir, 'pot2vai_' + str(best_score) + '_D.pth'))
                torch.save(classifier.state_dict(),
                           osp.join(args.snapshot_dir, 'pot2vai_' + str(best_score) + '_classifier.pth'))

                fh = open(acc_path, 'a')
                fh.write('iter:' + str(i_iter))
                fh.write(metrics.to_str(val_score))
                fh.close()

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'pot2vai_' + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'pot2vai_' + str(i_iter) + '_D.pth'))
            torch.save(classifier.state_dict(),
                       osp.join(args.snapshot_dir, 'pot2vai_' + str(i_iter) + '_classifier.pth'))

if __name__ == '__main__':
    main()

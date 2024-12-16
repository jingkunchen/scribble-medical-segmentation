import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator
from networks.net_factory_1 import net_factory
from utils import losses, metrics, ramps
from utils.gate_crf_loss import ModelLossSemsegGatedCRF
from val_2D import test_single_volume_cct, test_single_volume_ds, test_single_volume_matching

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from PIL import Image
import os



if torch.cuda.is_available():
    print("GPU is available.")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU found.")

folder = 1

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/WeaklySeg_pCE_matching_final_scribble_'+str(folder), help='experiment_name')
parser.add_argument('--fold', type=str,
                    default=folder, help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--model', type=str,
                    default='unet_matching_1', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=60000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=6,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
args = parser.parse_args()


def tv_loss(predication):
    min_pool_x = nn.functional.max_pool2d(
        predication * -1, (3, 3), 1, 1) * -1
    contour = torch.relu(nn.functional.max_pool2d(
        min_pool_x, (3, 3), 1, 1) - min_pool_x)
    # length
    length = torch.mean(torch.abs(contour))
    return length

def crf(original_image, mask_img):    
    original_image = original_image[0,:,:] *255
    original_image = Image.fromarray(original_image)
    original_image = original_image.convert('RGB')
    original_image = np.asarray(original_image)
    mask_img[mask_img >= 0.5] = 255
    mask_img[mask_img <0.5] = 1
    
    mask_img = Image.fromarray(mask_img)
    mask_img = mask_img.convert('RGB')
    mask_img=  np.asarray(mask_img)
    annotated_label = mask_img[:,:,0] + (mask_img[:,:,1]<<8) + (mask_img[:,:,2]<<16)

    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 2

    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)


    U = unary_from_labels(labels, n_labels, gt_prob=0.7 , zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #Run Inference for 10 steps 
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((original_image.shape[0],original_image.shape[1]))


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes).cuda()
    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]), fold=args.fold, sup_type=args.sup_type)
    db_val = BaseDataSets(base_dir=args.root_path, fold=args.fold, split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = losses.pDLoss(num_classes, ignore_index=4)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    alpha = 1.0
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch, onehot_label_batch,final_selection = sampled_batch['image'], sampled_batch['label'], sampled_batch['onehot_label'], sampled_batch['final_selection']
            volume_batch, label_batch, onehot_label_batch,final_selection = volume_batch.cuda(), label_batch.cuda(), onehot_label_batch.cuda() , final_selection.cuda()
            outputs, feature_layer, pixel_embedding, sematic_embedding = model(
                volume_batch, label_batch, onehot_label_batch, final_selection)
           
            outputs_soft1 = torch.softmax(outputs, dim=1)
            
            loss_ce = ce_loss(outputs, label_batch[:].long())
            beta = random.random() + 1e-10
            pseudo_supervision = torch.argmax(
                (outputs_soft1.clone().detach()), dim=1, keepdim=False)
            loss_pse_sup1 = dice_loss(outputs_soft1, pseudo_supervision.unsqueeze(1))

            loss_pse_sup = loss_pse_sup1 

           
            loss = loss_ce + 0.5 * loss_pse_sup
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/loss_pse_sup1', loss_pse_sup1, iter_num)
            writer.add_scalar('info/loss_pse_sup', loss_pse_sup, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            
            if wandb_logger:
                wandb.log({"loss_ce": loss_ce.item()})
                wandb.log({"loss_pse_sup": loss_pse_sup.item()})
                wandb.log({"loss": loss.item()})
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_ce: %f, alpha: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_ce.item(), alpha))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
                writer.add_image('train/pseudo_supervision',pseudo_supervision[1,None, ...] * 50, iter_num)
                writer.add_image('train/onehot_label_0',onehot_label_batch[1,None, :,:,0] * 50, iter_num)
                writer.add_image('train/onehot_label_1',onehot_label_batch[1,None, :,:,1] * 50, iter_num)
                writer.add_image('train/onehot_label_2',onehot_label_batch[1,None, :,:,2] * 50, iter_num)
                writer.add_image('train/onehot_label_3',onehot_label_batch[1,None, :,:,3] * 50, iter_num)
                writer.add_image('train/onehot_label_4',onehot_label_batch[1, None,:,:,4] * 50, iter_num)
                
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_matching(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                if wandb_logger:
                    wandb.log({"val dice": performance})
                    wandb.log({"val mean_hd95": mean_hd95})
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()
            
            if iter_num > 0 and iter_num % 500 == 0:
                if alpha > 0.01:
                    alpha = alpha - 0.01
                else:
                    alpha = 0.01

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.fold, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code',
    #                 shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)

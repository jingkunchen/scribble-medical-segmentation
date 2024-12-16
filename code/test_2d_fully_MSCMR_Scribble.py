import argparse
import os
import re
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

# from networks.efficientunet import UNet
from networks.net_factory import net_factory
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
patch_size_w = 480
patch_size_h = 480


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/MSCMRSegCVPR', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='MSCMRSegCVPR/WeaklySeg_pCE_sematic_matching_weight_1_1_final', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_matching', help='model_name')
parser.add_argument('--fold', type=str,
                    default='', help='fold')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--sup_type', type=str, default="scribble",
                    help='label')

def crf(original_image, mask_img):

    
    original_image = original_image*255
    original_image = Image.fromarray(original_image)
    original_image = original_image.convert('RGB')
    original_image = np.asarray(original_image)

    
    
    mask_img = mask_img.astype(np.uint8)
    mask_img = Image.fromarray(mask_img)
    mask_img = mask_img.convert('RGB')
    mask_img=  np.asarray(mask_img)
    # print("new_ mask_img:",mask_img.shape)
#     #Converting the annotations RGB color to single 32 bit integer
    annotated_label = mask_img[:,:,0] + (mask_img[:,:,1]<<8) + (mask_img[:,:,2]<<16)
    
#     # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 4
    
    #Setting up the CRF model
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    # get unary potentials (neg log probability)
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



def get_fold_ids(fold):
    all_cases_set = ["subject{0:d}_".format(i) for i in [2, 4, 6, 7, 9, 13, 14, 15, 18, 19, 20, 21, 22, 24, 25, 26, 27, 31, 32, 34, 37, 39, 42, 44, 45]]
    fold1_testing_set = [
            "subject{0:d}_".format(i) for i in [2, 4, 6, 7, 9]]
    fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]

    fold2_testing_set = [
            "subject{0:d}_".format(i) for i in [13, 14, 15, 18, 19]]
    fold2_training_set = [
            i for i in all_cases_set if i not in fold2_testing_set]

    fold3_testing_set = [
            "subject{0:d}_".format(i) for i in [20, 21, 22, 24, 25]]
    fold3_training_set = [
            i for i in all_cases_set if i not in fold3_testing_set]

    fold4_testing_set = [
            "subject{0:d}_".format(i) for i in [26, 27, 31, 32, 34]]
    fold4_training_set = [
            i for i in all_cases_set if i not in fold4_testing_set]

    fold5_testing_set = [
            "subject{0:d}_".format(i) for i in [37, 39, 42, 44, 45]]
    fold5_training_set = [
            i for i in all_cases_set if i not in fold5_testing_set]
    if fold == "fold1":
        return [fold1_training_set, fold1_testing_set]
    elif fold == "fold2":
        return [fold2_training_set, fold2_testing_set]
    elif fold == "fold3":
        return [fold3_training_set, fold3_testing_set]
    elif fold == "fold4":
        return [fold4_training_set, fold4_testing_set]
    elif fold == "fold5":
        return [fold5_training_set, fold5_testing_set]
    else:
        return "ERROR KEY"


def calculate_metric_percase(pred, gt, spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if np.count_nonzero(pred) == 0:
        print("pred:",np.count_nonzero(pred),np.unique(pred, return_counts=True))
        print("gt:",np.unique(gt, return_counts=True))
        pred[0][0]=True
    
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt, voxelspacing=spacing)
    #asd = 0
    hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
    return dice, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path +
                    "/MSCMRSegCVPR_training_volumes/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice_ = image[ind, :, :]
        x, y = slice_.shape[0], slice_.shape[1]
        slice = zoom(slice_, (patch_size_h / x, patch_size_w / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            #unet
            #out_main = net(input)
            #unet_matching
            out_main,_,_,_ = net(input,None,None,None)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size_h, y / patch_size_w), order=0)
            prediction[ind] = pred
    case = case.replace(".h5", "")
    org_img_path = "../data/MSCMRSegCVPR_training/{}_norm.nii.gz".format(case)
    org_img_itk = sitk.ReadImage(org_img_path)
    spacing = org_img_itk.GetSpacing()

    first_metric = calculate_metric_percase(
        prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
    second_metric = calculate_metric_percase(
        prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
    third_metric = calculate_metric_percase(
        prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.CopyInformation(org_img_itk)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.CopyInformation(org_img_itk)
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.CopyInformation(org_img_itk)
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    train_ids, test_ids = get_fold_ids(FLAGS.fold)
    all_volumes = os.listdir(
        FLAGS.root_path + "/MSCMRSegCVPR_training_volumes")
    image_list = []
    for ids in test_ids:
        new_data_list = list(filter(lambda x: re.match(
            '{}.*'.format(ids), x) != None, all_volumes))
        image_list.extend(new_data_list)
    snapshot_path = "../model/{}_{}/{}".format(
        FLAGS.exp, FLAGS.fold, FLAGS.sup_type)
    test_save_path = "../model/{}_{}/{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.fold, FLAGS.sup_type, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
       #snapshot_path, 'unet_best_model.pth')
       snapshot_path, 'unet_matching_best_model.pth')
        
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    first_total_ = []
    second_total_ = []
    third_total_ = []
    for case in tqdm(image_list):
        print(case)
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        first_total_.append(first_metric)
        second_total_.append(second_metric)
        third_total_.append(third_metric)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    return avg_metric, [first_total_, second_total_, third_total_]


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    total = []
    total_pp = []
    for i in [1, 2, 3, 4, 5]:
        FLAGS.fold = "fold{}".format(i)
        all_matrix, all_matrix_pp = Inference(FLAGS)
        for j in range(5):
            total_pp.append(np.array(all_matrix_pp)[:,j])
        total.append(all_matrix)
    
   
    total_pp = np.array(total_pp)
    print("total_pp:",total_pp.shape)
    print('class1:',np.around(np.mean(total_pp[:, 0, :],axis=0),3),np.around(np.std(total_pp[:, 0, :],axis=0),3))
    print('class2:',np.around(np.mean(total_pp[:, 1, :],axis=0),3),np.around(np.std(total_pp[:, 1, :],axis=0),3))
    print('class3:',np.around(np.mean(total_pp[:, 2, :],axis=0),3),np.around(np.std(total_pp[:, 2, :],axis=0),3))
    print('final:',np.around(np.mean(np.mean(total_pp,axis=1),axis=0),3),np.around(np.std(np.mean(total_pp,axis=1),axis=0),3))

    average = []
    for i in range(5):
        print("fold:",i+1, total[i], np.mean(np.array(total[i]), axis=0))
        average.append(np.mean(total[i], axis=0), )
    print('avergae_perfold:',np.mean(average, axis=0))

import itertools
import os
import random
import re
from glob import glob

import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import cv2
import math
from einops import einsum
import datetime
import time


def pseudo_label_generator_acdc(data, seed, beta=100, mode='bf'):
    from skimage.exposure import rescale_intensity
    from skimage.segmentation import random_walker
    if 1 not in np.unique(seed) or 2 not in np.unique(seed) or 3 not in np.unique(seed):
        pseudo_label = np.zeros_like(seed)
    else:
        markers = np.ones_like(seed)
        markers[seed == 4] = 0
        markers[seed == 0] = 1
        markers[seed == 1] = 2
        markers[seed == 2] = 3
        markers[seed == 3] = 4
        sigma = 0.35
        data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                                 out_range=(-1, 1))
        segmentation = random_walker(data, markers, beta, mode)
        pseudo_label = segmentation - 1
    return pseudo_label


class BaseDataSets_superpixel(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label"):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        train_ids, test_ids = self._get_fold_ids(fold)   
                
        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + "/ACDC_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)
                
        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        fold1_testing_set = [
            "patient{:0>3}".format(i) for i in range(1, 21)]
        fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]

        fold2_testing_set = [
            "patient{:0>3}".format(i) for i in range(21, 41)]
        fold2_training_set = [
            i for i in all_cases_set if i not in fold2_testing_set]

        fold3_testing_set = [
            "patient{:0>3}".format(i) for i in range(41, 61)]
        fold3_training_set = [
            i for i in all_cases_set if i not in fold3_testing_set]

        fold4_testing_set = [
            "patient{:0>3}".format(i) for i in range(61, 81)]
        fold4_training_set = [
            i for i in all_cases_set if i not in fold4_testing_set]

        fold5_testing_set = [
            "patient{:0>3}".format(i) for i in range(81, 101)]
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

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_slices/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_volumes/{}".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        scribble = h5f['scribble'][:]
        superpixel = h5f['superpixel'][:]
        superpixel_label = h5f['superpixel_label'][:]
        sample = {'image': image, 'label': label, 'scribble': scribble, 'superpixel': superpixel, 'superpixel_label': superpixel_label}
        if self.split == "train":
            if self.sup_type == "random_walker":
                label = pseudo_label_generator_acdc(image, h5f["scribble"][:])
            else:
                label = h5f[self.sup_type]
            sample = {'image': image, 'label': label, 'scribble': scribble, 'superpixel': superpixel, 'superpixel_label': superpixel_label}
            sample = self.transform(sample)            
        else:
            
            
            # superpixel_label2 = zoom(s
            # superpixel_label, (256 / (image.shape[0]*4), 256 / (image.shape[1]*4)), order=0)
    
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.uint8))

            scribble =  torch.from_numpy(scribble.astype(np.uint8))

            superpixel = torch.from_numpy(superpixel.astype(np.uint8))
        
            superpixel_label = torch.from_numpy(superpixel_label.astype(np.uint8))
            
           
            sample = {'image': image, 'label': label, 'scribble': scribble, 'superpixel': superpixel, 'superpixel_label': superpixel_label}
        sample["idx"] = idx
        return sample

class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label"):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        train_ids, test_ids = self._get_fold_ids(fold)
        
        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + "/ACDC_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        fold1_testing_set = [
            "patient{:0>3}".format(i) for i in range(1, 21)]
        fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]

        fold2_testing_set = [
            "patient{:0>3}".format(i) for i in range(21, 41)]
        fold2_training_set = [
            i for i in all_cases_set if i not in fold2_testing_set]

        fold3_testing_set = [
            "patient{:0>3}".format(i) for i in range(41, 61)]
        fold3_training_set = [
            i for i in all_cases_set if i not in fold3_testing_set]

        fold4_testing_set = [
            "patient{:0>3}".format(i) for i in range(61, 81)]
        fold4_training_set = [
            i for i in all_cases_set if i not in fold4_testing_set]

        fold5_testing_set = [
            "patient{:0>3}".format(i) for i in range(81, 101)]
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

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_slices/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_volumes/{}".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        superpixel_label =None
        sample = {'image': image, 'label': label}
        if self.split == "train":
            image = h5f['image'][:]
            if self.sup_type == "random_walker":
                label = pseudo_label_generator_acdc(image, h5f["scribble"][:])
            else:
                label = h5f[self.sup_type][:]

            sample = {'image': image, 'label': label, 'superpixel_label': superpixel_label}
            
            sample = self.transform(sample)
            
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            # superpixel_label = h5f['superpixel_label'][:]
            onehot_label = np.eye(5, dtype='int64')[label]
            sample = {'image': image, 'label': label, 'onehot_label': onehot_label,'final_selection': onehot_label}
        sample["idx"] = idx
        return sample


def random_rot_flip(image, label, superpixel_label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    #superpixel_label = np.rot90(superpixel_label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    #superpixel_label = np.flip(superpixel_label, axis=axis).copy()
    return image, label, superpixel_label

def random_rot_flip_superpixel(image, label, scribble, superpixel, superpixel_label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    scribble = np.rot90(scribble, k)
    superpixel = np.rot90(superpixel, k)
    superpixel_label = np.rot90(superpixel_label, k)
    
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    scribble = np.flip(scribble, axis=axis).copy()
    superpixel = np.flip(superpixel, axis=axis).copy()
    superpixel_label= np.flip(superpixel_label, axis=axis).copy()
    return image, label, scribble, superpixel, superpixel_label
   
def random_rotate(image, label, superpixel_label, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    #superpixel_label = ndimage.rotate(superpixel_label, angle, order=0,reshape=False, mode="constant", cval=0)
    return image, label, superpixel_label


def random_rotate_superpixel(image, label, scribble, superpixel, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    scribble = ndimage.rotate(scribble, angle, order=0,
                              reshape=False, mode="constant", cval=4)
    superpixel = ndimage.rotate(superpixel, angle, order=0,reshape=False)
    superpixel_label = ndimage.rotate(superpixel_label, angle, order=0,reshape=False, mode="constant", cval=cval)
    # print("superpixel:", np.unique(superpixel, return_counts=True))
    return image, label, scribble, superpixel, superpixel_label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, superpixel_label = sample['image'], sample['label'], sample['superpixel_label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        image, label, superpixel_label = random_rot_flip(image, label, superpixel_label)


        #if random.random() > 0.5:
        #    image, label, superpixel_label = random_rot_flip(image, label, superpixel_label)
        #
        #elif random.random() > 0.5:
        if 4 in np.unique(label):
                image, label, superpixel_label = random_rotate(image, label, superpixel_label, cval=4)
        else:
                image, label, superpixel_label = random_rotate(image, label, superpixel_label, cval=0)

        x, y = image.shape
        
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)#
        #superpixel_label = zoom(
        #    superpixel_label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        onehot_label = np.eye(5, dtype='int64')[label]
        point_0 = onehot_label[:,:,0].nonzero()
        zero_mask_0 = np.zeros((image.shape[0], image.shape[1]))
        if len(point_0[0]) >0:
            x_min = np.min(point_0[0])
            x_max = np.max(point_0[0])
            y_min = np.min(point_0[1])
            y_max = np.max(point_0[1])
            zero_mask_0[x_min:x_max, y_min:y_max] = 1
        
        point_1 = onehot_label[:,:,1].nonzero()
        zero_mask_1 = np.zeros((image.shape[0], image.shape[1]))
        if len(point_1[0]) >0:
            x_min = np.min(point_1[0])-5
            x_max = np.max(point_1[0])+5
            y_min = np.min(point_1[1])-5
            y_max = np.max(point_1[1])+5
            zero_mask_1[x_min:x_max, y_min:y_max] = 1
        
        point_2 = onehot_label[:,:,2].nonzero()
        zero_mask_2 = np.zeros((image.shape[0], image.shape[1]))
        if len(point_2[0]) >0:
            x_min = np.min(point_2[0])-5
            x_max = np.max(point_2[0])+5
            y_min = np.min(point_2[1])-5
            y_max = np.max(point_2[1])+5
            zero_mask_2[x_min:x_max, y_min:y_max] = 1
        
        point_3 = onehot_label[:,:,3].nonzero()
        zero_mask_3 = np.zeros((image.shape[0], image.shape[1]))
        if len(point_3[0]) >0:
            x_min = np.min(point_3[0])-5
            x_max = np.max(point_3[0])+5
            y_min = np.min(point_3[1])-5
            y_max = np.max(point_3[1])+5
            zero_mask_3[x_min:x_max, y_min:y_max] = 1
        
        
        final_selection = np.array([zero_mask_0, zero_mask_1,zero_mask_2,zero_mask_3])
        
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        
        onehot_label = torch.from_numpy(onehot_label.astype(np.float32))
        # final_selection = torch.from_numpy(final_selection.astype(np.float32))
        # superpixel_label = np.eye(4, dtype='int64')[superpixel_label.astype(np.int32)]
        # print('superpixel_label:', superpixel_label.shape)
        # final_selection = torch.from_numpy(superpixel_label.astype(np.float32)).permute(2,0,1)
        #mask = F.interpolate(onehot_label, size=[64,64], mode='bilinear', align_corners=True)
        
        sample = {'image': image, 'label': label, 'onehot_label': onehot_label, 'final_selection': final_selection}
        return sample

class RandomGenerator_superpixel(object):
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, sample):
        image, label, scribble, superpixel, superpixel_label = \
                sample['image'], sample['label'], sample['scribble'], sample['superpixel'], sample['superpixel_label']
        
        #without rotate        
        image, label, scribble, superpixel, superpixel_label= \
                    random_rot_flip_superpixel(image, label, scribble, superpixel, superpixel_label)
        # if random.random() > 0.5:
        #     image, label, scribble, superpixel= \
        #             random_rot_flip_superpixel(image, label, scribble, superpixel)
        # elif random.random() > 0.5:
        #     if 4 in np.unique(label):
        #         image, label, scribble, superpixel= \
        #             random_rotate_superpixel(image, label, scribble, superpixel, cval=4)
        #     else:
        #         image, label, scribble, superpixel = \
        #             random_rotate_superpixel(image, label, scribble, superpixel, cval=0)
        
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        scribble = zoom(
            scribble, (self.output_size[0] / x, self.output_size[1] / y), order=0)
      
        superpixel = zoom(
            superpixel, (self.output_size[0] / (x *4), self.output_size[1] / (y *4)), order=0)

        superpixel_label1 = zoom(
            superpixel_label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        superpixel_label2 = zoom(
            superpixel_label, (self.output_size[0] / (x*4), self.output_size[1] / (y*4)), order=0)
        superpixel_label = superpixel_label1
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        
        scribble =  torch.from_numpy(scribble.astype(np.uint8))
        
        superpixel = torch.from_numpy(superpixel.astype(np.uint8))
        
        superpixel_label = torch.from_numpy(superpixel_label.astype(np.uint8))
        
        superpixel_label2 = torch.from_numpy(superpixel_label2.astype(np.uint8))

        sample = {'image': image, 'label': label, 'scribble': scribble, 'superpixel': superpixel, 'superpixel_label': superpixel_label, 'superpixel_label2': superpixel_label2}
        
        
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

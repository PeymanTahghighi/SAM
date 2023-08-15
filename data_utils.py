from glob import iglob
from shutil import copyfile
from typing import Any
import pandas
import os
import cv2
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pickle
from glob import glob
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import config
from sklearn.model_selection import train_test_split

def visualize_annotations(img, annot, window_name = None):
    ret = deepcopy(img);
    for a in annot:
        cv2.circle(ret.squeeze(), (int(a[0]), int(a[1])), 5, (255, 255, 255), -1);

    return ret;
    

class HandDataset(Dataset):
    def __init__(self, img_annot_pair, transforms) -> None:
        super().__init__()
        self.img_annot_pair = img_annot_pair;
        self.resize = Resize(config.RESIZE);
        self.crop = RandomCrop();
        self.pre_processs_transform = A.Compose([A.Normalize()]);
    
    def __len__(self):
        return len(self.img_annot_pair);

    def __getitem__(self, index):
        img_annot_pair = pickle.load(open(self.img_annot_pair[index], 'rb'));
        landmarks = img_annot_pair[1];
        landmarks = [[i,landmarks[i], landmarks[i+1]] for i in range(0, len(landmarks), 2)];
        pair1, pair2 = self.crop({'image': img_annot_pair[0], 'landmarks': landmarks});
        pair1 = self.resize(pair1);
        pair2 = self.resize(pair2);

        #padd landmarks
        landmarks1 = np.array(pair1['landmarks']);
        landmarks1_pad = np.zeros((37,2));
        mask = np.zeros((37));
        landmarks1_pad[:landmarks1.shape[0],:] = landmarks1;
        landmarks1_pad = np.array([l[0]*config.RESIZE+l[1] for l in landmarks1_pad])

        landmarks2 = np.array(pair2['landmarks']);
        landmarks2_pad = np.zeros((37,2));
        landmarks2_pad[:landmarks2.shape[0],:] = landmarks2;
        landmarks2_pad = np.array([l[0]*config.RESIZE+l[1] for l in landmarks2_pad])
        mask[:landmarks2.shape[0]] = 1;


        image1 = self.pre_processs_transform(image = pair1['image'])['image'];
        image2 = self.pre_processs_transform(image = pair2['image'])['image'];

        return {'image': image1, 'landmarks': landmarks1_pad}, {'image': image2, 'landmarks': landmarks2_pad}, mask;

class Resize(object):
    def __init__(self, size):
        self.size = size;

    def __call__(self, sample):
        debug_annotations = False;
        image, landmarks = sample['image'], sample['landmarks'];
        h,w,_ = image.shape;
        scale_h = self.size / h;
        scale_w = self.size / w;
        scaled_img = A.Resize(self.size,self.size)(image = image)['image'];
        scaled_landmarks = [];
        for l in landmarks:
            scaled_landmarks.append([int(l[0]*scale_w), int(l[1]*scale_h)]);

        if debug_annotations:
            new_img_np = scaled_img.squeeze();
            scaled_landmarks = np.array(scaled_landmarks);
            val1 = new_img_np[scaled_landmarks[:,0], scaled_landmarks[:,1], :];
            new_img_np_big = new_img_np.reshape(400*400,3);
            scaled_landmarks_new = [l[0]*config.RESIZE+l[1] for l in scaled_landmarks]
            val2 = new_img_np_big[scaled_landmarks_new,:];
            for i in range(len(val1)):
                if val1[i][0] != val2[i][0]:
                    print('not');
            scaled_landmarks_new = [[int(np.floor(l/config.RESIZE)), l%config.RESIZE] for l in scaled_landmarks_new]
            for i in range(len(scaled_landmarks_new)):
                if scaled_landmarks_new[i][0] != scaled_landmarks[i][0] or scaled_landmarks_new[i][1] != scaled_landmarks[i][1]:
                    print('not');

            #val2 = new_img_np_big
            temp_landmarks = np.concatenate([np.expand_dims(np.array(scaled_landmarks)[:,0], axis = 1), np.expand_dims(np.array(scaled_landmarks)[:,1], axis = 1)], axis = 1);
            ret = visualize_annotations(new_img_np, temp_landmarks);
            cv2.imshow('i', ret);
            cv2.waitKey();

        return {'image': scaled_img, 'landmarks': scaled_landmarks};

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self):
        pass
    
    def crop(self, image, landmarks):
        h,w,_ = image.shape;
        size_h = np.random.randint(int(h*2/3),h);
        size_w = np.random.randint(int(w*2/3),w);
        start_h = np.random.randint(0,h-size_h);
        end_h = start_h+size_h;
        start_w = np.random.randint(0,w-size_w);
        end_w = start_w+size_w;
        cropped_img = image[start_h:end_h, start_w:end_w,:];
        annot_in_crop = [];
        for a in landmarks:
            if a[2]<end_h and a[2] > start_h and a[1] < end_w and a[1] > start_w:
                annot_in_crop.append([a[0], a[1]-start_w, a[2]-start_h]);
                

        return cropped_img, annot_in_crop;

    def __call__(self, sample):
        debug_annotations = False
        image, landmarks = sample['image'], sample['landmarks']

        cropped_img1, annot_in_crop1 = self.crop(image, landmarks);
        cropped_img2, annot_in_crop2 = self.crop(image, landmarks);
        annot_in_crop2_ids = np.array(annot_in_crop2)[:,0];
        matched_pairs_1 = [];
        matched_pairs_2 = [];
        #match pairs
        for a1 in annot_in_crop1:
            if a1[0] in annot_in_crop2_ids:
                matched_pairs_1.append(a1[1:]);
                matched_pairs_2.append(annot_in_crop2[list(annot_in_crop2_ids).index(a1[0])][1:]);

        if debug_annotations:
            
            ret1 = visualize_annotations(cropped_img1, [[a[0], a[1]] for a in matched_pairs_1], 'crop1');
            ret2 = visualize_annotations(cropped_img2, [[a[0], a[1]] for a in matched_pairs_2], 'crop2');
            fig, ax = plt.subplots(1,2);
            ax[0].imshow(ret1, cmap='gray');
            ax[1].imshow(ret2, cmap='gray');
            plt.show();
        
        return {'image': cropped_img1, 'landmarks': matched_pairs_1}, {'image': cropped_img2, 'landmarks': matched_pairs_2}
    
class ToTensor(object):
    def __init__(self) -> None:
        pass

    def __call__(self, sample) -> Any:
        image, landmarks = sample['image'], sample['landmarks'];

        pass

def cache_dataset():
    pd = pandas.read_csv('dataset/all.csv');
    imgs = list(pd['img']);
    for img_name in imgs:
        img = cv2.imread(f'dataset/{img_name}.jpg', cv2.IMREAD_COLOR);
        row = list(pd.iloc[imgs.index(img_name),1:]);
        pickle.dump([img.squeeze(), row], open(f'dataset/{img_name}.dmp', 'wb'));

def get_loader():
    train_transforms = transforms.Compose([RandomCrop(), Resize(1024)]);
    img_annot_pairs = glob('dataset/*.dmp');
    train_pair, test_pair = train_test_split(img_annot_pairs, test_size=0.2);
    train_loader = DataLoader(HandDataset(train_pair, train_transforms), config.BATCH_SIZE, False, num_workers=0, pin_memory=True);
    test_loader = DataLoader(HandDataset(test_pair, train_transforms), config.BATCH_SIZE, False, num_workers=0, pin_memory=True);

    return train_loader,test_loader;   
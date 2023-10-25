import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 

from PIL import Image
import cv2
import os 
import shutil
import random
import timm
import timm.optim
import timm.scheduler
from timm.data import ImageDataset, create_dataset, create_loader
from timm.data.transforms_factory import create_transform

from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchinfo import summary

from fastai.vision.all import *
from fastai.vision.all import load_learner
from fastai.vision.data import ImageDataLoaders
from fastai.metrics import accuracy, F1Score

warnings.filterwarnings('ignore')


image_size = 224
batch = 32

train_df = pd.read_csv('/home/dip_21/project/cloud/train.csv')
# train_df['id'] = train_df['id'].apply(lambda x : "/home/dip_21/project/cloud/images/train/" + x)

dls = ImageDataLoaders.from_df(train_df,
                               path='/home/dip_21/project/cloud/images/train',
                                    valid_pct=0.15,
                                    bs = batch ,
                                    shuffle_train = True,
                                    item_tfms=[ToTensor(),Resize(image_size)] ,
                                    batch_tfms=[*aug_transforms(do_flip=False,
                                                                flip_vert=False,
                                                                max_rotate=360,
                                                                p_affine=0.8,
                                                                max_warp=0.2),
                                                Normalize.from_stats(*imagenet_stats),
                                               ],
                                    seed = 123,
                                    )

learn = load_learner('/home/dip_21/project2/vit_kfold/vit_fold_0.pkl',cpu=False)

# preds = learn.predict('/home/dip_21/project/cloud/images/train/22f8406f96ef5fb476422428f99beea0.jpg')
# print(preds)


df_submission = pd.read_csv('/home/dip_21/project/cloud/submit.csv')
df_test=df_submission.copy()
df_test['id']='/home/dip_21/project/cloud/images/test/'+df_test['id']


test_dl = learn.dls.test_dl(df_test['id'])

preds, decoder = learn.get_preds(dl = test_dl)

labels = np.argmax(preds, 1).tolist()

print(len(labels))


df_test['label'] = labels
df_test['label'] = [dls.vocab[i] for i in df_test['label'] ]

df_test.to_csv('/home/dip_21/project/cloud/test.csv',index=False)



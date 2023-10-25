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
import wandb
warnings.filterwarnings('ignore')


image_size = 224
batch = 32


train_df = pd.read_csv('/home/dip_21/project/cloud/train.csv')
# train_df['id'] = train_df['id'].apply(lambda x : "/home/dip_21/project/cloud/images/train/" + x)
test_df = pd.read_csv('/home/dip_21/project/cloud/test.csv')

pseudo_df = pd.concat([train_df, test_df], axis=0)
pseudo_df = pseudo_df.reset_index(drop=True)

print(len(train_df))
print(len(test_df))
print(len(pseudo_df))

# Group by 'label' and count occurrences
# class_distribution = pseudo_df['label'].value_counts()

# # Plot the class distribution
# plt.figure(figsize=(10, 6))
# bars = plt.bar(class_distribution.index, class_distribution.values)

# plt.xlabel('Class Label')
# plt.ylabel('Frequency')
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center', color='black', size=10)
# plt.title('Class Distribution')
# plt.savefig('/home/dip_21/project2/code/plot/classdist_pseudo.jpg')


dls = ImageDataLoaders.from_df(pseudo_df,
                               path='/home/dip_21/project2/data',
                                    valid_pct=0.15,
                                    bs = batch ,
                                    shuffle_train = True,
                                    item_tfms=[ToTensor(),Resize(image_size)] ,
                                    batch_tfms=[*aug_transforms(do_flip=False,
                                                                flip_vert=True,
                                                                max_rotate=360,
                                                                p_affine=0.8,
                                                                max_warp=0.2),
                                                Normalize.from_stats(*imagenet_stats),
                                               ],
                                    seed = 123,
                                    )

dls.train.show_batch(max_n=30)
plt.savefig('/home/dip_21/project2/code/plot/aug_pseudo.jpg')

# loss_func2 = CrossEntropyLossFlat(weight=class_weights)
save_cb = SaveModelCallback(monitor='valid_loss')

# Create a list of callbacks
callbacks = [save_cb] 
model_name = "vit_large_patch16_224"


learn = vision_learner(dls, model_name,
                       path='/home/dip_21/project2/vit_pseudo/',
                       cbs=[ShowGraphCallback()] ,
                       metrics=[accuracy])  # metrics=[accuracy]
                     #    #,WandbCallback()     force_download=True, 
learn.to_fp16()
# learn.model = torch.nn.DataParallel(learn.model)
wandb.init(
    project="NephoReg"

)
learn.lr_find()
learn.fine_tune(10,cbs=callbacks)
learn.export('new_vit_pseudo_2.pkl')
learn.validate()
learn.show_results()
plt.savefig('/home/dip_21/project2/code/plot/new_result_pseudo2.jpg')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
plt.savefig('/home/dip_21/project2/code/plot/new_conf_pseudo2.jpg')
interp.print_classification_report()
interp.plot_top_losses(20)
plt.savefig('/home/dip_21/project2/code/plot/new_loss_pseudo2.jpg')



import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from PIL import Image

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
from fastai.vision.data import ImageDataLoaders
from fastai.metrics import accuracy, F1Score
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import gc
import wandb
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

avail_pretrained_models = timm.list_models('vit_large_patch16_224*')
print(f'{len(avail_pretrained_models)} {avail_pretrained_models[:]}')

image_size = 224
batch = 32

train_df = pd.read_csv('/home/dip_21/project/cloud/train.csv')
train_df['id'] = train_df['id'].apply(lambda x : "/home/dip_21/project/cloud/images/train/" + x)

strat_kfold = MultilabelStratifiedKFold(n_splits=10, random_state=42, shuffle=True)
train_df['fold'] = -1
for i, (_, test_index) in enumerate(strat_kfold.split(train_df.id.values, train_df.iloc[:,1:].values)):
    train_df.iloc[test_index, -1] = i
print(train_df.head())


    
# dls = ImageDataLoaders.from_df(train_df,
#                                     path='/home/dip_21/project/cloud/images/train',
#                                     valid_pct=0.15,
#                                     bs = batch ,
#                                     shuffle_train = True,
#                                     item_tfms=[ToTensor(),Resize(image_size)] ,
#                                     batch_tfms=[*aug_transforms(do_flip=False,
#                                                                 flip_vert=True,
#                                                                 max_rotate=360,
#                                                                 p_affine=0.8,
#                                                                 max_warp=0.2),
#                                                 Normalize.from_stats(*imagenet_stats),
#                                               ],
#                                     seed = 786,
#                                     )

item_tfms=[ToTensor(),Resize(image_size)]
batch_tfms=[*aug_transforms(do_flip=False,
                            flip_vert=True,
                            max_rotate=360,
                            p_affine=0.8,
                            max_warp=0.2),
            Normalize.from_stats(*imagenet_stats),
          ]

splitter = TrainTestSplitter(0.15, stratify=train_df["label"],random_state=42,shuffle=True)
def get_data(fold=0):
  return DataBlock(blocks=(ImageBlock,CategoryBlock),
                      get_x=ColReader(0),
                      get_y=ColReader(1),
                      item_tfms=item_tfms,
                      splitter=IndexSplitter(train_df[train_df.fold == fold].index),
                      batch_tfms=batch_tfms
                            ).dataloaders(train_df, bs=32)
wandb.init(
      project="NephoReg"

  )
model_name = "vit_large_patch16_224"
save_cb = SaveModelCallback(monitor='valid_loss')
for i in range(10):
  dls = get_data(i)
  learn = vision_learner(dls, model_name,
                        path='/home/dip_21/project2/vit_kfold/',
                        cbs=[save_cb] ,
                        metrics=[accuracy])  # metrics=[accuracy]
                      #    #,WandbCallback()     force_download=True, 
  learn.to_fp16()
  # learn.model = torch.nn.DataParallel(learn.model)
  learn.fine_tune(10)
  learn.export(f'vit_fold_{i}.pkl')
  learn.validate()
  learn.show_results()
  plt.savefig(f'/home/dip_21/project2/code/plot/new_result_{i}.jpg')
  interp = ClassificationInterpretation.from_learner(learn)
  interp.plot_confusion_matrix()
  plt.savefig(f'/home/dip_21/project2/code/plot/new_conf_{i}.jpg')
  interp.print_classification_report()
  interp.plot_top_losses(10)
  plt.savefig(f'/home/dip_21/project2/code/plot/new_loss_{i}.jpg')
  del learn
  torch.cuda.empty_cache()
  gc.collect()

  print(f'fold {i} finish')







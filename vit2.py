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



avail_pretrained_models = timm.list_models('vit_large_patch16_224*')
print(f'{len(avail_pretrained_models)} {avail_pretrained_models[:]}')

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
                                                                flip_vert=True,
                                                                max_rotate=360,
                                                                p_affine=0.8,
                                                                max_warp=0.2),
                                                Normalize.from_stats(*imagenet_stats),
                                               ],
                                    seed = 123,
                                    )

dls.train.show_batch(max_n=30)
plt.savefig('/home/dip_21/project2/code/plot/aug.jpg')

# loss_func2 = CrossEntropyLossFlat(weight=class_weights)
save_cb = SaveModelCallback(monitor='valid_loss')

# Create a list of callbacks
callbacks = [save_cb] 
model_name = "vit_large_patch16_224"


learn = vision_learner(dls, model_name,
                       path='/home/dip_21/project2/vit_new/',
                       cbs=[ShowGraphCallback()] ,
                       metrics=[accuracy])  # metrics=[accuracy]
                     #    #,WandbCallback()     force_download=True, 
learn.to_fp16()
# learn.model = torch.nn.DataParallel(learn.model)
wandb.init(
    project="NephoReg"

)
learn.fine_tune(10,cbs=callbacks)
learn.export('new_vit.pkl')
learn.validate()
learn.show_results()
plt.savefig('/home/dip_21/project2/code/plot/new_result.jpg')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
plt.savefig('/home/dip_21/project2/code/plot/new_conf.jpg')
interp.print_classification_report()
interp.plot_top_losses(10)
plt.savefig('/home/dip_21/project2/code/plot/new_loss.jpg')





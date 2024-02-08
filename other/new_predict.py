from fastai.vision.all import load_learner,PILImage

learn = load_learner('/home/dip_21/project2/vit_new/new_vit.pkl',cpu=False)
img = PILImage.create('/home/dip_21/project/cloud/images/train/22f8406f96ef5fb476422428f99beea0.jpg')
def perdict(img,learn):
    pred,pred_idx,pred_prob = learn.predict(img)
    print(f' class : {pred} prob : {pred_prob[pred_idx]}')
perdict(img,learn)

import torch
from torch.autograd import Variable
import tools.utils as utils
import tools.dataset as dataset
from PIL import Image
from collections import OrderedDict
import argparse
import numpy as np
# from pytorch_grad_cam import CAM
# from pytorch_grad_cam.utils.image import  show_cam_on_image
import cv2
import time
from models.model import MODEL
from models.transformer.Constants import UNK,PAD,BOS,EOS,PAD_WORD,UNK_WORD,BOS_WORD,EOS_WORD
# from pytorchgradcam import CAM, GuidedBackpropReLUModel
# from pytorchgradcam.utils.image import show_cam_on_image, \
#                                          deprocess_image, \
#                                          preprocess_image
import os,sys,pdb
# from skimage.color import gray2rgb


model_path = sys.argv[1]
img_path = sys.argv[2]
img_name = img_path.split('.')[0].split('/')[-1]

alphabet = '0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z ! " \' # $ % & ( ) * + , - . / : ; < = > ? @ [ \\ ] _ ` ~'
n_bm = 5
imgW = 100
imgH = 32
nclass = len(alphabet.split(' '))
MODEL = MODEL(n_bm, nclass, dec_layer=3, LR=True)
# Total_params = 0

if torch.cuda.is_available():
    MODEL = MODEL.cuda()

print('loading pretrained model from %s' % model_path)
state_dict = torch.load(model_path)
MODEL_state_dict_rename = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "") # remove `module.`
    MODEL_state_dict_rename[name] = v
MODEL.load_state_dict(MODEL_state_dict_rename, strict=True)

# layer_name = list(state_dict.keys())
# layer_name = list(MODEL.state_dict(state_dict).keys())
# for i in range(len(layer_name)):
#     print(i, ' ', layer_name[i])

for p in MODEL.parameters():
    p.requires_grad = False
    # mulValue = np.prod(p.size())
    # Total_params += mulValue

# for name in MODEL.state_dict():
#     print(name)
# print(MODEL.state_dict()['encoder.conv2'].size())


MODEL.eval()
#print(Total_params)

converter = utils.strLabelConverterForAttention(alphabet, ' ')
transformer = dataset.resizeNormalize((imgW, imgH))
image = Image.open(img_path).convert('RGB')
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)
print(image.size())
text = torch.LongTensor(1 * 5)
length = torch.IntTensor(1)
text = Variable(text)
length = Variable(length)

max_iter = 35
t, l = converter.encode('0'*max_iter)
utils.loadData(text, t)
utils.loadData(length, l)

# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)
# start.record()
# preds = MODEL(image, length, text, text, test=True, cpu_texts='')[0]
preds = MODEL(image, length, text, test=True, cpu_texts='')[0]

# da = MODEL(image, length, text, text, test=True, cpu_texts='')[4]
# end.record()
# torch.cuda.synchronize()
# print(start.elapsed_time(end))

pred = converter.decode(preds.data, length.data + 5)
pred = pred.split(' ')[0]
print('################# Answer: '+pred)
# image = image.squeeze(0)
# print(image.size())
# image = image.cpu().numpy().astype(np.uint8)
# image = image.transpose(1, 0, 2)
# image = image.transpose
# image = image.cpu().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
# input_tensor = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# H, W, C = image.shape
# print('H', H)
# print('w', W)
# print('c', C)
# da = np.array(da)
# print(da.shape)
# da = da.squeeze(1)
# da = da.squeeze(0)
# att = da.cpu().numpy()
# att_map = cv2.resize(att, (W, H))
# att_max = att_map.max()
# att_map /= att_max
# att_map *= 255
# att_map = att_map.astype(np.uint8)
# heatmap = cv2.applyColorMap(att_map, cv2.COLORMAP_JET)
#
# show_attention = image.copy()
# show_attention = cv2.addWeighted(heatmap, 2, show_attention, 200, 0)
# # _att_map = np.zeros(dtype=np.uint8, shape=[H, W, 3])
# # _att_map[:, :, -1] = (att_map * 255).astype(np.uint8)
# #
# # show_attention = cv2.addWeighted(image, 0.5, _att_map, 10, 2)
# cv2.imwrite('123_cam.jpg', show_attention)
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import random
import torch
toPIL = transforms.ToPILImage()
toTensor = transforms.ToTensor()

def heatmap(img, scores):

    try:
        i = 0
        input_img = '/home/zdz/Ubuntubeifen/dataset/image_release/benchmark_cleansed/IC15/test/word_1093.png'
        filename = 'out' + str(scores.size(1)) + '.png'
        filename = 'out' + str(i) + '.png'

        scores = scores.permute(1, 0, 2)

        scores = scores.cpu().numpy()

        scores = np.mean(scores, axis=1)[i].reshape(8, 25)
        img = cv2.imread(input_img)
        h, w, _ = img.shape
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        scores = np.uint8(255 * scores)
        scores = scores.astype(np.uint8)

        heatmap = cv2.applyColorMap(cv2.resize(scores, (w, h)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite('./pic/c/' + str(i) + str(random.randint(0, 100)) + filename, result)

    except:
        pass







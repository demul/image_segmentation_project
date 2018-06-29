import numpy as np
# from imageio import imread
# import matplotlib as plt
from matplotlib.pyplot import imshow, show, hist


def make_colormap_from_label(img_batch) :
    class_color_list = []
    img_batch = np.reshape(img_batch, [img_batch.shape[0] * img_batch.shape[1] * img_batch.shape[2], img_batch.shape[3]])
    for i in range (img_batch.shape[0]):
        pixel = list(img_batch[i])
        if pixel in  class_color_list:
            continue
        else :
            class_color_list.append(pixel)

    with open('colormap.txt', 'w') as wf :
        for i, color in enumerate(class_color_list):
            wf.write('Label: ' + str(i) + ' Color: ' + str(color)+'\n')


def make_dict_from_colormap() :
    class_color_list = []
    with open('colormap.txt', 'r') as rf :
        data = rf.readlines()
        for _, line in enumerate(data):
            idx = line.find('[')
            color = line[idx+1:-2]
            color = color.split(',')
            tmpmap = map (lambda x : eval(x), color)
            RGBcolor = np.array(list(tmpmap), dtype=np.uint8)
            class_color_list.append(RGBcolor)
    ########## 검은색(배경)과
    ########## 흰회색(경계선)은 제외해준다.
    ########## PASCAL VOC의 경우 배경, 경계선 색의 인덱스는 0,1이다.
    del class_color_list[0:2]

    return class_color_list


def intersection_over_union(pred, label_batch) :
    pred = np.squeeze(pred)
    label = np.argmax(label_batch, axis=3)

    pred_bg = pred ==0
    label_bg = label ==0

    fg =  ~(pred_bg | label_bg)

    intersection =np.sum((pred==label) & fg)

    union = np.sum(~(pred_bg & label_bg))

    return intersection / union



class masker :
    def __init__(self):
        class_color_list = make_dict_from_colormap()
        class_color_list.insert(0, np.array([0, 0, 0], dtype=np.uint8))  # 아까 만들 때 검은색(0)은 제거해줬으므로 해당하는 인덱스에 다시 삽입
        self.class_color_list = class_color_list


    def make_mask_from_label(self, label_batch):
        label_batch = np.squeeze(label_batch.astype(np.uint8))

        mask = np.take(np.asarray(self.class_color_list), label_batch, axis=0)
        return mask
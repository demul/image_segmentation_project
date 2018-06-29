import os
import numpy as np
from imageio import imread
from skimage.transform import resize
from matplotlib.pyplot import imshow, hist, show, figure
import util

class ImgLoader :

    def __init__(self, dataset='coco'):
        if dataset == 'pascal' :
            pascal_root = 'C:\VOC2012'
            seg_folder = 'ImageSets/Segmentation'  # Annotation 위치 (txt)

            origin_img = 'JPEGImages'  # 원본 데이터 위치 (jpg)
            class_img = 'SegmentationClass'  # ground truth 위치 (png)

            self.seg_image_annotation = os.path.join(pascal_root, seg_folder)
            self.seg_image_origin = os.path.join(pascal_root, origin_img)
            self.seg_image_class = os.path.join(pascal_root, class_img)
        else:
            coco_root = 'C:\coco'
            seg_folder = 'SimpleAnnotation'  # Annotation 위치 (txt)

            origin_img = 'images'  # 원본 데이터 위치 (jpg)
            class_img = 'SegmentationClass'  # ground truth 위치 (png)

            self.seg_image_annotation = os.path.join(coco_root, seg_folder)
            self.seg_image_origin = os.path.join(coco_root, origin_img)
            self.seg_image_class = os.path.join(coco_root, class_img)


        self.origin_path = None
        self.class_path = None
        self.class_color_list = None


    def load_name_list(self, train_or_val):
        with open(self.seg_image_annotation + '/' + train_or_val + '.txt', 'r') as f:
            lines = f.readlines()
            # 파일명 양 끝의 공백과 특수문자 제거
            # list 단위로 한꺼번에 실행하기 위해 다음 함수 사용.
            ## map(리스트에 함수 적용하여 매핑)
            ## lambda(임시함수의 주소 반환)
            data_lines = map(lambda x: x.strip(), lines)
            path2origin = map(lambda x: os.path.join(self.seg_image_origin, x) + '.jpg', data_lines)
            data_lines = map(lambda x: x.strip(), lines)
            path2class = map(lambda x: os.path.join(self.seg_image_class, x) + '.png', data_lines)
            # map -> list 변환 후 리턴
            origin_img_list = list(path2origin)
            class_img_list = list(path2class)
            return origin_img_list, class_img_list

    def load_img_list(self, file_path, batch_size, batch_count):
        img_list = []

        for i in range(batch_count, batch_count+batch_size):
            file = file_path[i]
            img = imread(file)
            img_list.append(img)
        return img_list # 테스트용으로 일부러 작게함. 크게하면 터지기도 하고...


    def calculate_size(self,img_list):
        h_list = []
        w_list =[]
        for img in img_list :
            h, w, _ = img.shape
            h_list.append(h)
            w_list.append(w)

        h_max = max(h_list)
        h_min = min(h_list)
        w_max = max(w_list)
        w_min = min(w_list)
        return h_max, h_min, w_max, w_min


    def make_batch_resize(self, img_list, height, width, interpolation=1):
        batch=np.empty((
            len(img_list),
            height,
            width,
            3
        ), dtype=np.float32)

        for idx, img in enumerate(img_list) :
            if (len(img.shape)<3) :                    ###MS-COCO에는 흑백도 섞여있다; ㅡㅡ
                img=np.tile(img, (3,1,1))              ###그냥 3 채널로 복사해버려서 가짜 흑백 이미지를 만들자. 새 차원을 추가하면서 복제할려면 이런식으로 해야 한다.
                img=np.transpose(img, (1,2,0))         ###맨 앞차원이 늘어나게 되므로 맨 앞차원을 맨 뒷차원으로 전치시켜줘야한다.

            batch[idx] = resize(img[:, :, :3], (height, width, 3), order=interpolation) *255 #png파일이라 R,G,B,alpha의 4차원 데이터이므로 alpha차원을 제거
        ################################################################################################
        ####### class image의 경우 픽셀값이 소수가 되는 것을 방지하기 위해 NN으로 보간해야 한다!########
        ################################################################################################
        # skimage.transform.resize(img, output_size, order)
        # order=0: Nearest - neighbor
        # order=1: Bi - linear(default)
        # order=2: Bi - quadratic
        # order=3: Bi - cubic
        # order=4: Bi - quartic
        # order=5: Bi - quintic
        return batch


    def make_label_batch(self, img_batch,):
        newbatch = np.empty((
            img_batch.shape[0],
            img_batch.shape[1],
            img_batch.shape[2],
            len(self.class_color_list)+1 #배경의 차원 하나를 추가해 줄 것이므로!
        ), dtype=np.float32)

        for i in range (img_batch.shape[0]):
            label_fg = np.zeros([img_batch.shape[1], img_batch.shape[2]], dtype=np.bool)
            class_img = img_batch[i, :, :, :].astype(np.uint8)
            for j, color in enumerate(self.class_color_list):
                label = np.all(class_img == color, axis=2)
                label_fg |= label
                newbatch[i, :, :, j+1] = label.astype(np.float32)

            label_bg = ~label_fg
            newbatch[i, :, :, 0] = label_bg.astype(np.float32)

        return newbatch


    def run(self, train_or_val) :
        self.origin_path, self.class_path = self.load_name_list(train_or_val)
        self.class_color_list = util.make_dict_from_colormap()
        #hmax, hmin, wmax, wmin=calculate_size(class_img_list)

        if not os.path.isfile('colormap.txt'):
            print('There is no Color Map. Making Color Map.')
            self.make_colormap()


    def nextbatch(self, batch_size, itr):

        origin_img_list = self.load_img_list(self.origin_path, batch_size, itr)
        class_img_list = self.load_img_list(self.class_path, batch_size, itr)
##########################각종 Agumentation 기법을 여기넣으면 좋을듯###############################
        input_batch = self.make_batch_resize(origin_img_list, 320, 320, 1)
        class_batch = self.make_batch_resize(class_img_list, 320, 320, 0)
        class_label_batch = self.make_label_batch(class_batch)
##################################################################################################
        return input_batch, class_label_batch

    def nextbatch_for_inference(self, batch_size, itr):

        origin_img_list = self.load_img_list(self.origin_path, batch_size, itr)
        class_img_list = self.load_img_list(self.class_path, batch_size, itr)
        ##########################각종 Agumentation 기법을 여기넣으면 좋을듯###############################
        input_batch = self.make_batch_resize(origin_img_list, 320, 320, 1)
        class_batch = self.make_batch_resize(class_img_list, 320, 320, 0)
        ##################################################################################################
        return input_batch, class_batch

    def make_colormap(self):
        label_img_list = []
        for file in self.class_path :
            label_img = imread(file)
            label_img_list.append(label_img)

        class_batch = self.make_batch_resize(label_img_list, 320, 320, 0)
        util.make_colormap_from_label(class_batch)
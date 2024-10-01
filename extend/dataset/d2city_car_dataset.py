from torch.utils.data import DataLoader,Dataset
from skimage import io,transform
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np
import PIL.Image as Image
import sys
import json
from pycocotools.coco import COCO as COCO_old
import torch.nn.functional as F
from extend.utils.homo_affine import homography, homography_batch
from tqdm import tqdm
from extend.utils.iou import compute_iou_tensor

from torchvision.utils import save_image

from extend.utils.image_io import open_img

import os.path as osp


'''
please download d2city dataset, cut videos into frames and select 1 image per 30 frames.

'''


class D2City_car(Dataset): 
    def __init__(self, dataType='train', img_size=(720,720), use_num=None, use_cuda=True, only_return_name=False):

        self.img_size = img_size
        self.use_num = use_num
        self.use_cuda = use_cuda
        self.only_return_name = only_return_name


        annFile = '../common_data/d2city/'+dataType+'_label_convert_mod30.json'
        # annFile = '../common_data/coco/annotations/person_keypoints_'+dataType+'2017.json'
        # self.img_prefix= '../common_data/coco/'+dataType+'2017'
        self.img_prefix = '../common_data/d2city/'

        self.coco = COCO_old(annFile)
        AnnoIds_big_car_allselected_np_path = 'extend/dataset/d2city_train_car_use_norepeat.npy'

        self.save_mask_dir = '../common_data/d2city/mask'


        AnnoIds_big_car_allselected_np = np.load(AnnoIds_big_car_allselected_np_path)
        AnnoIds_big_car_allselected_list = list(AnnoIds_big_car_allselected_np)
        AnnoIds_big_car_allselected = [int(annoid) for annoid in AnnoIds_big_car_allselected_list]

        attr_dict = [
            {"supercategory": "none", "id": 1, "name": "car"},
            {"supercategory": "none", "id": 2, "name": "bus"},
            {"supercategory": "none", "id": 3, "name": "truck"},
            {"supercategory": "none", "id": 4, "name": "person rider"},
            {"supercategory": "none", "id": 5, "name": "bike"},
            {"supercategory": "none", "id": 6, "name": "motor"},
        ]

        self.frcn_select_annoid_list = AnnoIds_big_car_allselected[:self.use_num]

    def __len__(self):
        return len(self.frcn_select_annoid_list)
    
    def __getitem__(self, index):

        anno_id = self.frcn_select_annoid_list[index] 
        anns = self.coco.loadAnns(anno_id)
        imgId = anns[0]['image_id']

        img = self.coco.loadImgs(imgId)[0]
        img_name = img['file_name']
        area_anno = np.array(anns[0]['area'])
        img_name = img_name.split('/')[1]

        if self.only_return_name:
            return img_name

        img_pil = Image.open(os.path.join(self.img_prefix, img['file_name'])).convert('RGB')

        w = img_pil.width
        h = img_pil.height

        trans = transforms.Compose([
            transforms.Resize((self.img_size[0], int(self.img_size[0] / h * w))),
            transforms.ToTensor()]
        )
        image = trans(img_pil)


        # image resize and pad  AND  coordinate shift


        bbox_anno = np.array(anns[0]['bbox']) # "bbox": [x1,y1,width,height]
        bbox_x1y1x2y2 = np.array([
            bbox_anno[0], 
            bbox_anno[1], 
            bbox_anno[0] + bbox_anno[2],
            bbox_anno[1] + bbox_anno[3]
        ])

        bbox_x1y1x2y2[[0,2]] = self.img_size[0] / h * bbox_x1y1x2y2[[0,2]]
        bbox_x1y1x2y2[[1,3]] = int(self.img_size[0] / h * w) / w * bbox_x1y1x2y2[[1,3]]

        # bbox to center of square
        x_left = (bbox_x1y1x2y2[0]+bbox_x1y1x2y2[2])/2 - self.img_size[0]/2
        x_right = (bbox_x1y1x2y2[0]+bbox_x1y1x2y2[2])/2 + self.img_size[0]/2
        if x_left < 0:
            img_crop = image[:,:,0:self.img_size[0]]
            bbox_x1y1x2y2 = bbox_x1y1x2y2
        elif x_right > self.img_size[1] / h * w:
            img_crop = image[:,:,int(self.img_size[0] / h * w)-self.img_size[0]:int(self.img_size[0] / h * w)]
            bbox_x1y1x2y2[[0,2]] = bbox_x1y1x2y2[[0,2]] - (int(self.img_size[0] / h * w)-self.img_size[0])
        else:
            img_crop = image[:,:,int(x_left):int(x_right)]
            bbox_x1y1x2y2[[0,2]] = bbox_x1y1x2y2[[0,2]] - x_left

        
        image_nopad = img_crop
        box_area = anns[0]['area']
        self.save_mask_dir


        save_mask_path = os.path.join(self.save_mask_dir, img_name.split('.')[0]+'.png')

        if osp.exists(save_mask_path):
            mask = open_img(save_mask_path)
            mask_1c = mask[0]
            box_area_after_resize = (mask_1c>0).sum().item()
            area_mask = mask

        else:

            box_area_after_resize = (bbox_x1y1x2y2[2] - bbox_x1y1x2y2[0]) * (bbox_x1y1x2y2[3] - bbox_x1y1x2y2[1])
            assert box_area_after_resize >= 0
            box_area_after_resize = float(box_area_after_resize)


            area_mask = torch.zeros(3, self.img_size[0], self.img_size[1])
            area_mask[:, int(bbox_x1y1x2y2[1]):int(bbox_x1y1x2y2[3]), int(bbox_x1y1x2y2[0]):int(bbox_x1y1x2y2[2])] = 1



        keypoints = 0
        

        class_id = torch.tensor([2]).item() # car in coco
        image_real_mask = torch.ones(3, self.img_size[0], self.img_size[1])

        return image_nopad, keypoints, img_name, bbox_x1y1x2y2, class_id, \
            box_area_after_resize, image_real_mask, area_mask
    




def main():
    import os
    import sys
    import copy
    from mmdet.apis import init_detector, inference_detector, show_result_pyplot, get_Image_ready, show_result_pyplot_save



    config_file = './configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = '../common_data/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    # config_file = './configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
    # checkpoint_file = '../common_data/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
    # config_file = './configs/yolo/yolov3_d53_mstrain-416_273e_coco.py'
    # checkpoint_file = '../common_data/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth'
    # config_file = './configs/ssd/ssd512_coco.py'
    # checkpoint_file = '../common_data/ssd512_coco_20200308-038c5591.pth'

    model = init_detector(config_file, checkpoint_file, device='cpu').cuda().eval()


    mean = torch.Tensor([123.675, 116.28, 103.53]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
    std = torch.Tensor([1., 1., 1.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()

    img_frcn_1344 = get_Image_ready(model, 'd2city/val/0008/0a65321e9bd2e5b098e93f8c34bd304a_1.jpg')

    # 750*1333---->768*1344


    batch_size = 16

    # use_img_size = 800

    data = D2City_car(dataType='train', img_size=(1080,1920), use_cuda=True, only_return_name=False, use_num=None)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False) #使用DataLoader加载数据
    sum = 0

    anno_use_all = []
    print(len(data))
    for i_batch, batch_data in enumerate(tqdm(dataloader)):
        img_batch = batch_data[0].float().cuda()
        anno_id_batch = batch_data[1].float().cuda()

        img_name_batch = batch_data[2]
        bbox_batch = batch_data[3].float()
        object_area_batch = batch_data[4].float().cuda()

        batch_size_now = img_batch.shape[0]
        

        img_batch_255 = img_batch* 255.
        img_batch_nom = (img_batch_255 - mean) /std

        pad_shape = img_frcn_1344['img_metas'][0][0]['pad_shape']
        pad_shape_h = pad_shape[0]
        pad_shape_w = pad_shape[1]
        target_size = img_frcn_1344['img_metas'][0][0]['img_shape'][:2]

        img_batch_nom_rsz = F.interpolate(img_batch_nom, target_size, mode='bilinear', align_corners=False)

        img_batch_nom_pad = torch.cuda.FloatTensor(img_frcn_1344['img'][0].shape).fill_(0).repeat(batch_size_now,1,1,1)
        img_batch_nom_pad[:,:,:target_size[0], :target_size[1]] = img_batch_nom_rsz
        img_new = copy.deepcopy(img_frcn_1344)
        img_new['img'][0] = img_batch_nom_pad
        img_new['img_metas'][0] = img_frcn_1344['img_metas'][0] * batch_size_now

        with torch.no_grad():
            model_output = model(return_loss=False, rescale=True,  **img_new)


        
        cls_conf_thr = 0.8
        iou_thr = 0.7
        

        for i in range(batch_size_now):

            car_predict_now = model_output[i][2] 

            if car_predict_now.shape[0] == 0:
                continue
            
            car_predict_now = car_predict_now[car_predict_now[:,4]>cls_conf_thr]

            car_predict_now = torch.from_numpy(car_predict_now)
            car_predict_now = car_predict_now[:,:4]

            bbox_i = bbox_batch[i].repeat(car_predict_now.shape[0],1)
            iou = compute_iou_tensor(bbox_i, car_predict_now)

            if not (iou<iou_thr).all():
                anno_id_i = anno_id_batch[i]
                anno_use_all.append(int(anno_id_i))
            # print(iou)
    
    
    a=np.array(anno_use_all)
    np.save('d2city_train_car_use_norepeat.npy', a)  


    print(len(anno_use_all))





def read(img_path):
    img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
    return img

def show(img):
    img_pil = transforms.ToPILImage()(img.squeeze().detach().cpu())
    img_pil.show()

def save(img, save_path):
    img_pil = transforms.ToPILImage()(img.squeeze().detach().cpu())
    img_pil.save(save_path)
        
if __name__=='__main__':
    main()

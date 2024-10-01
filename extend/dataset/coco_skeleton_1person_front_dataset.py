from torch.utils.data import DataLoader,Dataset
from skimage import io,transform
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np
import PIL.Image as Image
from pycocotools.coco import COCO as COCO_old
import torch.nn.functional as F
from ..utils.homo_affine import *
from tqdm import tqdm


'''
please download coco dataset.

'''

class COCO_skeleton_1person_front(Dataset):
    def __init__(self, dataType='val', img_size=800, 
        use_num=None, start_num=None, end_num=None,
        use_cuda=False, only_return_name=False):


        self.img_size = img_size


        self.use_cuda = use_cuda
        self.only_return_name = only_return_name
        

        # dataType = 'train'
        newly_generate = False


        # annFile = '../common_data/coco/annotations/person_keypoints_'+dataType+'2017.json'
        # self.coco = COCO_old(annFile)
        self.img_prefix= '../common_data/coco/'+dataType+'2017'

        save_dir = 'extend/dataset'
        save_name = dataType+'_full_demand_1_person_frcn.npy'
        save_path = os.path.join(save_dir, save_name)
        if newly_generate == False and os.path.exists(save_path):
            print(save_path, 'exists, use it.')
            # 读取
            front_select_list=np.load(save_path)
            front_select_list=front_select_list.tolist()

            annFile = '../common_data/coco/annotations/person_keypoints_'+dataType+'2017.json'
            self.coco = COCO_old(annFile)


        else:

            annFile = '../common_data/coco/annotations/person_keypoints_'+dataType+'2017.json'
            
            # initialize COCO api for instance annotations
            self.coco = COCO_old(annFile)

            catIds = self.coco.getCatIds(catNms=['person'])
            

            imgIds = self.coco.getImgIds(catIds=catIds)


            AnnoIds = self.coco.getAnnIds(catIds=catIds, iscrowd=False)
            print('new generate!')
            front_select_list = []
            for anno_id in tqdm(AnnoIds):
                ann = self.coco.loadAnns(anno_id)[0]
                if 'segmentation' in ann or 'keypoints' in ann:
                    datasetType = 'instances'
                elif 'caption' in ann:
                    datasetType = 'captions'
                else:
                    raise Exception('datasetType not supported')
                if datasetType == 'instances':
                    if 'keypoints' in ann and type(ann['keypoints']) == list:
                            # turn skeleton into zero-based index
                            sks = np.array(self.coco.loadCats(ann['category_id'])[0]['skeleton'])-1
                            kp = np.array(ann['keypoints'])
                            x = kp[0::3]
                            y = kp[1::3]
                            v = kp[2::3]
                            accept = True
                            
                            # head 
                            if not torch.all(v[[0,1,2]]>0):
                                accept = False; continue
                            if not x[1]>x[0]>x[2] :
                                accept = False; continue

                            # belly
                            if not torch.all(v[[5,6,11,12]]>0):
                                accept = False; continue
                            if not (x[5]>x[6] and x[11]>x[12]):
                                accept = False; continue

                            # area
                            imgId = ann['image_id']
                            img = self.coco.loadImgs(imgId)[0]
                            object_area = ann['area']
                            image_area = img['height']*img['width']
                            if not object_area / image_area >0.1:
                                accept = False; continue

                            # up arm
                            if not torch.all(v[[6,8,5,7]]>0):
                                accept = False; continue
                            
                            # down arm
                            if not torch.all(v[[8,10,7,9]]>0):
                                accept = False; continue
                            
                            # up leg
                            if not torch.all(v[[11,12,13,14]]>0):
                                accept = False; continue

                            # down leg
                            if not torch.all(v[[13,14,15,16]]>0):
                                accept = False; continue


                            img_annos = self.coco.getAnnIds(imgIds=imgId)
                            if len(img_annos) > 1:
                                accept = False; continue
                            
                            person_num = 0
                            for img_anno_id in img_annos:
                                img_anno = self.coco.loadAnns(img_anno_id)
                                if img_anno[0]['category_id'] == 1:
                                    person_num += 1
                            if person_num > 1:
                                accept = False; continue
                            print(len(img_annos), person_num)

                            if accept:
                                front_select_list.append(anno_id)


            a=np.array(front_select_list)
            np.save(save_path,a) 

        self.front_select_annoid_list = front_select_list
        if use_num is not None:
            self.front_select_annoid_list = self.front_select_annoid_list[:use_num]
        elif start_num is not None and end_num is not None:
            self.front_select_annoid_list = self.front_select_annoid_list[start_num:end_num]




    def __len__(self):
        return len(self.front_select_annoid_list)
    
    def __getitem__(self, index):

        anno_id = self.front_select_annoid_list[index] 
        anns = self.coco.loadAnns(anno_id)
        imgId = anns[0]['image_id']

        img = self.coco.loadImgs(imgId)[0]
        img_name = img['file_name']
        area_anno = np.array(anns[0]['area'])

        mask = self.coco.annToMask(anns[0])
        mask = torch.from_numpy(mask).float()
        mask = mask.repeat(3,1,1)

        if self.only_return_name:
            return img_name

        image = transforms.ToTensor()(Image.open('%s/%s' % (self.img_prefix, img['file_name'])).convert('RGB'))


        # image resize and pad  AND  coordinate shift
        w = image.shape[-1]
        h = image.shape[-2]
        keypoints = np.array(anns[0]['keypoints'])
        keypoints_3 = keypoints.reshape(-1,3)

        bbox_anno = np.array(anns[0]['bbox']) # "bbox": [x1,y1,width,height]
        bbox_x1y1x2y2 = np.array([
            bbox_anno[0], 
            bbox_anno[1], 
            bbox_anno[0] + bbox_anno[2],
            bbox_anno[1] + bbox_anno[3]
        ])

        if self.use_cuda:
            image_pad = torch.cuda.FloatTensor(3, self.img_size, self.img_size).fill_(0.5)
            image_real_mask = torch.cuda.FloatTensor(3, self.img_size, self.img_size).fill_(0)
            area_mask = torch.cuda.FloatTensor(3, self.img_size, self.img_size).fill_(0)
        else:
            image_pad = torch.ones(3, self.img_size, self.img_size).fill_(0.5)
            image_real_mask = torch.zeros(3, self.img_size, self.img_size)
            area_mask = torch.zeros(3, self.img_size, self.img_size)
        

        if w >= h: 
            target_size = [int(self.img_size/w*h), self.img_size]
            image_resize = F.interpolate(image.unsqueeze(0), target_size, mode='bilinear', align_corners=False).squeeze()
            area_mask_resize = F.interpolate(mask.unsqueeze(0), target_size, mode='bilinear', align_corners=False).squeeze()
            _y_start = int((self.img_size - target_size[0])/2)
            image_pad[:, _y_start:_y_start+target_size[0]] = image_resize

            # x
            keypoints_3[:, 0] = keypoints_3[:, 0]/w*self.img_size
            bbox_x1y1x2y2[[0,2]] = bbox_x1y1x2y2[[0,2]]/w*self.img_size
            # y
            keypoints_3[:, 1] = keypoints_3[:, 1]/w*self.img_size + _y_start
            bbox_x1y1x2y2[[1,3]] = bbox_x1y1x2y2[[1,3]]/w*self.img_size + _y_start

            # area
            resize_ratio = 1 / w * self.img_size
            area_real = area_anno*resize_ratio*resize_ratio

            # real img mask
            image_real_mask[:, _y_start:_y_start+target_size[0], :] = 1

            # area_mask
            area_mask[:, _y_start:_y_start+target_size[0], :] = area_mask_resize


        else:   
            target_size = [self.img_size, int(self.img_size/h*w)]
            image_resize = F.interpolate(image.unsqueeze(0), target_size, mode='bilinear', align_corners=False).squeeze()
            area_mask_resize = F.interpolate(mask.unsqueeze(0), target_size, mode='bilinear', align_corners=False).squeeze()
            _x_start = int((self.img_size - target_size[1])/2)
            image_pad[:, :, _x_start:_x_start+target_size[1]] = image_resize

            # x
            keypoints_3[:, 0] = keypoints_3[:, 0]/h*self.img_size + _x_start
            bbox_x1y1x2y2[[0,2]] = bbox_x1y1x2y2[[0,2]]/h*self.img_size + _x_start
            # y
            keypoints_3[:, 1] = keypoints_3[:, 1]/h*self.img_size
            bbox_x1y1x2y2[[1,3]] = bbox_x1y1x2y2[[1,3]]/h*self.img_size

            # area
            resize_ratio = 1 / h * self.img_size
            area_real = area_anno*resize_ratio*resize_ratio

            # real img mask
            image_real_mask[:, :, _x_start:_x_start+target_size[1]] = 1

            # area_mask
            area_mask[:, :, _x_start:_x_start+target_size[1]] = area_mask_resize

        
        keypoints = keypoints_3.reshape(-1)

        
        return image_pad, keypoints, img_name, bbox_x1y1x2y2, torch.tensor([0]), area_real, image_real_mask, area_mask






def main():
    import os
    import sys
    print(os.getcwd())
    print(sys.path)
    print(os.path.exists('22.png'))


    data = COCO_skeleton_1person_front(dataType='train', img_size = 800, use_cuda=False, only_return_name=False)
    dataloader = DataLoader(data, batch_size=1, shuffle=False) 
    sum = 0
    for i_batch, batch_data in enumerate(tqdm(dataloader)):
        sum = sum + 1
    print(sum)
    exit(0)




    # patches ready to train
    need_imgs = [
            'left_up_arm',
            'left_down_arm',

            'right_up_arm',
            'right_down_arm',

            'left_up_leg',
            'left_down_leg',

            'right_up_leg',
            'right_down_leg',

            'belly',
            'mask'
        ] 

    skeletons_num_dict = {
            'left_up_arm':[5,7],
            'left_down_arm':[7,9],

            'right_up_arm':[6,8],
            'right_down_arm':[8,10],

            'left_up_leg':[11,13],
            'left_down_leg':[13,15],

            'right_up_leg':[12,14],
            'right_down_leg':[14,16],

            'belly':[5,6,11,12],
            'mask':[0,1,2]
        }

    patch_layer = Image.open('refer_img.png').convert('RGB')
    patch_layer = transforms.ToTensor()(patch_layer)

    # patch_list = [patch for i in range(len(need_imgs))]

    # # patch_list[3] = None
    # patch_list[5].requires_grad_(True)

    # patch_dict = dict(zip(need_imgs, patch_list))


    # mask prepare
    mask_dir = './person_mask/'
    mask_name_list = os.listdir(mask_dir)
    mask_path_list = [os.path.join(mask_dir, mask_name) for mask_name in mask_name_list]

    mask_list = [read(mask_path) for mask_path in mask_path_list]

    mask_list = [torch.where(mask>0, torch.ones_like(mask), torch.zeros_like(mask)) for mask in mask_list]

    mask_dict = dict(zip(mask_name_list, mask_list))
    # ['person_chest.png', 'person_dleg1.png', 'person_dleg2.png', 'person_face.png', 'person_hand1.png', 'person_hand2.png', 'person_mask0.png', 'person_mask3.png', 'person_mask4.png', 'person_mask5.png', 'person_uleg1.png', 'person_uleg2.png']
    new_mask_dict = {}

    new_mask_dict['belly'] = mask_dict['person_chest.png']
    new_mask_dict['mask'] = mask_dict['person_face.png']
    new_mask_dict['left_up_arm'] = mask_dict['person_hand2.png']
    new_mask_dict['right_up_arm'] = mask_dict['person_hand1.png']
    new_mask_dict['left_up_leg'] = mask_dict['person_uleg2.png']
    new_mask_dict['right_up_leg'] = mask_dict['person_uleg1.png']
    new_mask_dict['left_down_leg'] = mask_dict['person_dleg2.png']
    new_mask_dict['right_down_leg'] = mask_dict['person_dleg1.png']

    move_range = 0.04
    

    data = COCO_skeleton_1person_front(img_size = 800)
    dataloader = DataLoader(data, batch_size=1, shuffle=False) 
    for i_batch, batch_data in enumerate(dataloader):
        image, kp = batch_data


        patch_dict, small_mask_dict = get_patch_dict_rand(patch_layer, new_mask_dict, move_range)
        # image = image[0]
        # kp = kp[0]
        # sks = sks[0]
        image = image
        kp = kp


        print(i_batch) 
        if i_batch < 70:
            continue
            print()

        


        
        patched_image = image.detach().clone()
        masked_mask = torch.zeros_like(image)



        x = kp[:,0::3].float()
        y = kp[:,1::3].float()
        v = kp[:,2::3]



        if patch_dict['belly'] != None:
            a5s = torch.stack([x[:,5], y[:,5]], dim=1)
            a6s = torch.stack([x[:,6], y[:,6]], dim=1)
            a11s = torch.stack([x[:,11], y[:,11]], dim=1)
            a12s = torch.stack([x[:,12], y[:,12]], dim=1)
            v5_6_11_12s = torch.all(v[:,[5,6,11,12]]>0,dim=1)

            # # test
            # v5_6_11_12s[0] = False
            # a5s[0] = 0
            # a6s[0] = 0
            # a11s[0] = 0
            # a12s[0] = 0


            patched_image, _mask = _render_belly_patch_batch(
                patched_image, 
                patch_dict['belly'],
                small_mask_dict['belly'],
                a5s,a6s,a11s,a12s, 
                v5_6_11_12s)

            masked_mask += _mask
            # save(patched_image[0], 'b0.png')
            # save(patched_image[1], 'b1.png')
            # save(patched_image[2], 'b2.png')
            

        # use a for circle to write this 

        for key in need_imgs[:-2]:
            if key in patch_dict and patch_dict[key] != None:
                if 'arm' in key:
                    width = torch.sqrt((x[:,5]-x[:,6]).pow(2)+(y[:,5]-y[:,6]).pow(2))/3
                    width_key_num1 = 5
                    width_key_num2 = 6
                elif 'leg' in key:
                    width = torch.sqrt((x[:,11]-x[:,12]).pow(2)+(y[:,11]-y[:,12]).pow(2))*2/3
                    width_key_num1 = 11
                    width_key_num2 = 12
                else:
                    print('no leg or arm in name! plz check')
                    raise 

                up_point_num = skeletons_num_dict[key][0]
                down_point_num = skeletons_num_dict[key][1]

                part_patch = patch_dict[key]
                part_mask = small_mask_dict[key]
                up_point = torch.stack([x[:,up_point_num], y[:,up_point_num]], dim=1)
                down_point = torch.stack([x[:,down_point_num], y[:,down_point_num]], dim=1)
                value_flag = torch.all(v[:,[width_key_num1,width_key_num2,up_point_num,down_point_num]]>0,dim=1)
                patched_image, _mask = _render_limb_patch_batch(patched_image, part_patch, part_mask, up_point, down_point, width, value_flag) # 为了函数的通用化 需要指定宽度
                masked_mask += _mask



def _render_belly_patch_batch(image, belly_patch, belly_mask, a5s,a6s,a11s,a12s, v5_6_11_12s):
    batch_size_now = image.shape[0]
    belly_patch_pad = torch.zeros_like(image)
    belly_mask_pad = torch.zeros_like(image)

    belly_patch = belly_patch.unsqueeze(0)
    belly_mask = belly_mask.unsqueeze(0)

    _y_start = int(image.shape[-2]/2-belly_patch.shape[-2]/2)
    _y_end = _y_start + belly_patch.shape[-2]
    _x_start = int(image.shape[-1]/2-belly_patch.shape[-1]/2)
    _x_end = _x_start + belly_patch.shape[-1]
    belly_patch_pad[:, :,
        _y_start : _y_end,
        _x_start : _x_end,
    ] = belly_patch
    belly_mask_pad[:, :,
        _y_start : _y_end,
        _x_start : _x_end,
    ] = belly_mask

    x2s = torch.Tensor([
        [_x_end, _y_start],
        [_x_start, _y_start],
        [_x_start, _y_end],
        [_x_end, _y_end],
    ]).repeat(batch_size_now,1,1)


    c5_6s = ( a5s + a6s )/2
    c11_12s = ( a11s + a12s )/2



    x1s = torch.stack([
        a5s,  # left shoulder
        a6s,  # right shoulder
        a6s + (c11_12s - c5_6s),  # right hip
        a5s + (c11_12s - c5_6s)   # left hip
    ],dim=1)

    # judge by v5_6_11_12s
    # the False also takes affine, but from x2s to x2s
    # this is for preventing no solution in homography()
    x1s = torch.where(v5_6_11_12s.unsqueeze(-1).unsqueeze(-1), x1s, x2s)


    theta_line = homography(x1s[0], x2s[0])

    theta_line = homography_batch(x1s, x2s)  # (b, 8, 1)

    theta_line = theta_line[:,:,0]  # (b, 8)

    a11,a12,a13,a21,a22,a23,_,_ = [theta_line[:,i] for i in range(8)]
    w = image.shape[-1]
    h = image.shape[-2]
    b11 = a11
    b12 = a12*h/w
    b13 = a13/(w/2)+a11+h/w*a12-1
    b21 = a21*w/h
    b22 = a22
    b23 = a21*w/h+a22+a23/(h/2)-1

    # theta = torch.Tensor([
    #     [b11, b12, b13],
    #     [b21, b22, b23]
    #     ])

    theta = torch.stack([
        torch.stack([b11, b12, b13],dim=-1),
        torch.stack([b21, b22, b23],dim=-1),
        ], dim=1)

    grid = F.affine_grid(theta, image.size(), align_corners=False)
    belly_patch_affine = F.grid_sample(belly_patch_pad, grid, align_corners=False)
    belly_mask_affine = F.grid_sample(belly_mask_pad, grid, align_corners=False)

    patched_ = torch.where(belly_mask_affine>0, belly_patch_affine, image)
    masked_ = torch.where(belly_mask_affine>0, belly_mask_affine, torch.zeros_like(belly_mask_affine))

    patched_varify = torch.where(v5_6_11_12s.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), patched_, image)
    masked_varify = torch.where(v5_6_11_12s.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), masked_, torch.zeros_like(masked_))

    return patched_varify, masked_varify


def _render_limb_patch_batch(image, limb_patch, limb_mask, up_point, down_point, width, value_flag):
    batch_size_now = image.shape[0]

    limb_patch_pad = torch.zeros_like(image)
    limb_mask_pad = torch.zeros_like(image)

    limb_patch = limb_patch.unsqueeze(0)
    limb_mask = limb_mask.unsqueeze(0)

    _y_start = int(image.shape[-2]/2-limb_patch.shape[-2]/2)
    _y_end = _y_start + limb_patch.shape[-2]
    _x_start = int(image.shape[-1]/2-limb_patch.shape[-1]/2)
    _x_end = _x_start + limb_patch.shape[-1]
    limb_patch_pad[:, :,
        _y_start : _y_end,
        _x_start : _x_end,
    ] = limb_patch
    limb_mask_pad[:, :,
        _y_start : _y_end,
        _x_start : _x_end,
    ] = limb_mask


    x2s = torch.Tensor([
        [_x_end, _y_start],
        [_x_start, _y_start],
        [_x_start, _y_end],
        [_x_end, _y_end],
    ]).repeat(batch_size_now,1,1)




    direction_vector_batch = down_point - up_point



    orthogonal_vector_batch = torch.where(
        (direction_vector_batch[:,1] == 0).unsqueeze(-1),
        torch.Tensor([0,1]).unsqueeze(0).repeat(batch_size_now,1),
        torch.stack([torch.Tensor([1]*batch_size_now), - (direction_vector_batch[:,0]/direction_vector_batch[:,1])], dim=-1))


    orthogonal_vector_norm_batch = orthogonal_vector_batch/ torch.sqrt(orthogonal_vector_batch[:,0].pow(2)+orthogonal_vector_batch[:,1].pow(2)).unsqueeze(-1)
    orthogonal_vector_with_width_batch = orthogonal_vector_norm_batch * width.unsqueeze(-1) /2

    def cross_product(v1,v2):
        return v1[0]*v2[1]-v2[0]*v1[1]
    def cross_product_batch(v1,v2):
        return v1[:,0]*v2[:,1]-v2[:,0]*v1[:,1]

    orthogonal_vector_with_width_batch = torch.where(
        (cross_product_batch(direction_vector_batch, orthogonal_vector_with_width_batch) < 0).unsqueeze(-1),
        -orthogonal_vector_with_width_batch,
        orthogonal_vector_with_width_batch
    )

    
    a1 = up_point - orthogonal_vector_with_width_batch
    a2 = up_point + orthogonal_vector_with_width_batch
    a3 = down_point + orthogonal_vector_with_width_batch
    a4 = down_point - orthogonal_vector_with_width_batch

    

    x1s = torch.stack([
        a1,
        a2,
        a3,
        a4
    ],dim=1)


    
    # judge by value_flag
    # the False also takes affine, but from x2s to x2s
    # this is for preventing no solution in homography()
    x1s = torch.where(value_flag.unsqueeze(-1).unsqueeze(-1), x1s, x2s)

    ###########################


    theta_line = homography_batch(x1s, x2s)  # (b, 8, 1)

    theta_line = theta_line[:,:,0]  # (b, 8)

    a11,a12,a13,a21,a22,a23,_,_ = [theta_line[:,i] for i in range(8)]
    w = image.shape[-1]
    h = image.shape[-2]
    b11 = a11
    b12 = a12*h/w
    b13 = a13/(w/2)+a11+h/w*a12-1
    b21 = a21*w/h
    b22 = a22
    b23 = a21*w/h+a22+a23/(h/2)-1



    theta = torch.stack([
        torch.stack([b11, b12, b13],dim=-1),
        torch.stack([b21, b22, b23],dim=-1),
        ], dim=1)
    ###########################


    grid = F.affine_grid(theta, image.size(), align_corners=False)
    limb_patch_affine = F.grid_sample(limb_patch_pad, grid, align_corners=False).squeeze()
    limb_mask_affine = F.grid_sample(limb_mask_pad, grid, align_corners=False).squeeze()

    patched_ = torch.where(limb_mask_affine>0, limb_patch_affine, image)
    masked_ = torch.where(limb_mask_affine>0, limb_mask_affine, torch.zeros_like(limb_mask_affine))


    patched_varify = torch.where(value_flag.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), patched_, image)
    masked_varify = torch.where(value_flag.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), masked_, torch.zeros_like(masked_))

    return patched_varify, masked_varify

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

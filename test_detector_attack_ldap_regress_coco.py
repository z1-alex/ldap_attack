import os
os.environ["OMP_NUM_THREADS"] = "3"
os.environ["MKL_NUM_THREADS"] = "3"
import cv2
cv2.setNumThreads(3)

from extend.dataset.coco_skeleton_1person_front_dataset import COCO_skeleton_1person_front
from extend.dataset.d2city_car_dataset import D2City_car

from extend.attack_method.ldap_paper import LDAPatch

from mmdet.structures.bbox import bbox_overlaps
from extend.utils.detection_success_judge import get_success_flag


from torch.utils.data import DataLoader

from torchvision.utils import save_image

from extend.model.detector_adv import init_detector_adv, save_detect_from_image, get_pred
import torch

from extend.model.model_names import get_model_name_scope_by_name


import random
import os
import numpy as np
import argparse
import json


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed_everything(0)

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet attack model')
    parser.add_argument('--model', type=str, help='the model to be attacked')
    parser.add_argument('--dataset', type=str, default='coco', help='the dataset to be attacked')
    parser.add_argument('--attackmethod', type=str, default='ldap', help='the attack method')
    parser.add_argument('--save_dir', type=str, help='the directory to save the files')
    parser.add_argument('--data_interval', type=int, default=1, help='the interval of data to be attacked')
    parser.add_argument('--max_step', type=int, default=2000, help='the max step of attack')


    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    save_dir = args.save_dir
    dataset_type = 'train'

    
    ##############################
    #     data
    ##############################
    if args.dataset == 'coco':
        img_size = (500, 500)
        dataset = COCO_skeleton_1person_front(dataType=dataset_type, img_size = img_size[0], only_return_name=False)
    elif args.dataset == 'didi':
        img_size = (720, 720)
        dataset = D2City_car(dataType=dataset_type, img_size = img_size, only_return_name=False)
    else:
        raise NotImplementedError
    data_interval = args.data_interval

    ##############################
    #     model
    ##############################

    target_model = args.model
    device = 'cuda:0'
    attack_model = get_model_name_scope_by_name(target_model)
    attack_name, attack_scope = attack_model
    attack_model, attack_resize_transform, attack_pad_transform, attack_inferencer = init_detector_adv(attack_name, attack_scope, device=device)
    attack_model = attack_model.to(device)
    attack_model.eval()
    model_plus = (attack_model, attack_resize_transform, attack_pad_transform, attack_inferencer)


    max_step = args.max_step


    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i_batch, batch_data in enumerate(dataloader):


        if i_batch % data_interval != 0:
            continue

        img, kp, img_name, gt_bbox, gt_label, object_area, image_real_mask, area_mask = batch_data

        img_name = img_name[0]

        # save a temp file to avoid repeat attack
        os.makedirs(os.path.join(save_dir, 'result'), exist_ok=True)
        save_det_result_path = os.path.join(save_dir, 'result', img_name.replace('.jpg', '.json'))
        if os.path.exists(save_det_result_path):
            print(f'{save_det_result_path} exists, skip')
            continue
        
        # if image has been saved skip
        save_image_dir = os.path.join(save_dir, 'image')
        if not os.path.exists(save_image_dir):
            os.makedirs(save_image_dir)
        saved_image_name_list = os.listdir(save_image_dir)
        if any([img_name.replace('.jpg', '_') in x for x in saved_image_name_list]):
            print(f'{img_name} exists, skip')
            continue

        print(f'Attack {img_name} start')
        with open(save_det_result_path, 'w') as f:
            f.write(' ')




        img = img[0].cuda()
        image_real_mask = image_real_mask[0].cuda()
        area_mask = area_mask[0].cuda()
        kp = kp[0].cuda()
        object_area = object_area[0].cuda().float()

        gt_bbox = gt_bbox[0].cuda().float()
        gt_label = gt_label[0].cuda().long()

        attacker = LDAPatch(img, gt_bbox, device='cuda')

        attacker.init_optimizer()

        for step in range(max_step):

            adv_img = attacker.get_adv_image(step=step)
            image_bchw_rgb_01 = adv_img[None].clamp(0, 1)

            bbox_pred, label_pred = get_pred(
                model_plus=model_plus,
                image_bchw_rgb_01=image_bchw_rgb_01,
                device=device,
            )

            bbox_pred = bbox_pred[0]
            label_pred = label_pred[0]

            iou_thr = 0.45
            iou = bbox_overlaps(bbox_pred[:,:4], gt_bbox[None])
            iou_mask = (iou>iou_thr).sum(dim=1)

            target_class = gt_label
            cls_mask = (label_pred == target_class).float()

            ###################################3
            # regression attack !
            # x1y1x2y2 box
                    
            if (gt_bbox[0] + gt_bbox[2])/2 > img.shape[-1]/2:
                # right become righter
                attack_direction = 1
            else:
                # left become lefter
                attack_direction = -1
            x_c = (bbox_pred[:, 0] + bbox_pred[:, 2])/ 2
            all_bbox_regress_loss = (iou_mask * cls_mask  * iou.detach() * x_c).sum() * attack_direction


            extra_loss = attacker.get_extra_loss(step=step)


            if step % 10 == 0:
                print(step, 
                'det_loss:',
                round(float(all_bbox_regress_loss),3), 
                'wh_loss:',
                round(float(extra_loss),3),
                'best_m:',
                round(float(attacker.best_mask_region),3),
                'best_m_r:',
                round(float(attacker.best_mask_region/ object_area) ,3),
                )


            det_success = get_success_flag(bbox_pred, label_pred, gt_bbox, gt_label)
            now_attack_area = attacker.get_use_patch_area()
            if now_attack_area < attacker.best_mask_region and not det_success:
                attacker.best_mask_region = now_attack_area
                attacker.best_mask = attacker.mask
                attacker.best_adv_image = adv_img

            total_loss = all_bbox_regress_loss + extra_loss
            total_loss.backward()
            attacker.optimize_step(step=step)
      

        save_dir = save_dir

        use_patch_area_rate = (attacker.best_mask_region / object_area).item()
        adv_image = attacker.best_adv_image


        adv_image_name = img_name.replace('.jpg', '') +'_' + str(use_patch_area_rate) + '.png'
        save_path = os.path.join(save_dir, 'image', adv_image_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_image(adv_image, save_path)

        save_vis_path = os.path.join(save_dir, 'visual', adv_image_name.replace('.png', '.jpg'))
        os.makedirs(os.path.dirname(save_vis_path), exist_ok=True)
        image_bchw_rgb_01 = adv_image[None].clamp(0, 1)
        save_detect_from_image(
            model_plus=model_plus,
            image_bchw_rgb_01=image_bchw_rgb_01,
            result_save_path=save_vis_path,
        )

        with open(save_det_result_path, 'w') as f:
            json_data = {
                'use_patch_area_rate': use_patch_area_rate,
            }

            try:
                bbox_pred_batch, label_pred_batch = get_pred(
                    model_plus=model_plus,
                    image_bchw_rgb_01=image_bchw_rgb_01,
                    device=device,
                )
                result_save = torch.cat([bbox_pred_batch[0], label_pred_batch[0][:,None]], dim=-1)
                result_save = result_save.tolist()
                json_data['det_result'] = result_save
            except:
                pass

            json.dump(json_data, f)

        print(f'Attack success, save to {save_path}, use patch area rate: {use_patch_area_rate}')
    print('Done')


if __name__ == '__main__':
    main()
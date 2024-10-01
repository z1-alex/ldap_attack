

import mmdet

from mmdet.structures.bbox import bbox_overlaps
import os.path as osp
import torch
import numpy as np
import copy
from mmcv.image import tensor2imgs
import mmcv
import torch.nn.functional as F
from torchvision.utils import save_image
from mmdet.apis import DetInferencer
from mmdet.structures import DetDataSample
from .differentiable_transforms import diff_resize, diff_pad, diff_pad_letter
import tempfile

from extend.utils.image_io import open_img

grounding_prompt = 'vehicle . car . sports car . sedan . SUV . jeep'


def get_gt_bbox(mask, device):
    assert mask.dim() == 2
    _yx = mask.nonzero()
    x1 = _yx[:,1].min()
    x2 = _yx[:,1].max()
    y1 = _yx[:,0].min()
    y2 = _yx[:,0].max()
    gt_box = torch.tensor([x1,y1,x2,y2], device=device).float()
    return gt_box


def init_detector_adv(model_name, scope, device='cuda:0'):

    inferencer = DetInferencer(model_name, scope=scope, device=device)
    model = inferencer.model
    model = model.to(device)
    model.eval()


    has_resize = False
    has_pad = False
    resize_transform = None
    pad_transform = None

    for transform in inferencer.pipeline.transforms:
        trans_class_name = transform.__class__.__name__
        if 'pad' in trans_class_name.lower():
            has_pad = True
            pad_transform = transform
        
        if 'resize' in trans_class_name.lower() and not 'letter' in trans_class_name.lower():
            has_resize = True
            resize_transform = transform

        if 'letter' in trans_class_name.lower():
            has_letter_pad = True
            pad_transform = transform
              


    return model, resize_transform, pad_transform, inferencer



def get_data_feed(image_bchw_rgb_01, resize_transform, pad_transform, grounding_prompt):
    
    img_tensor = image_bchw_rgb_01[:] # RGB
    img_tensor = img_tensor[:,[2,1,0],:,:]
    img_tensor255 = img_tensor * 255  # (3, H, W) BGR 0-255


    results = {}
    results['img'] = img_tensor255
    results = diff_resize(results, resize_transform)

    # img_tensor255 = results['img']
    # final_img_shape = results['img_shape']
    scale_factor = results['scale_factor']

    # pad
    if pad_transform is not None:
        if not 'letter' in pad_transform.__class__.__name__.lower():
            results = diff_pad(results, pad_transform)
        else:
            results = diff_pad_letter(results, pad_transform)
        pad_shape = results['pad_shape']


    img_tensor255 = results['img']
    final_img_shape = results['img_shape']
    pad_param = results.get('pad_param', None)

    img_shape = img_tensor.shape[2:]
    img_shape = tuple(img_shape)
    img_meta = dict(img_shape=final_img_shape,
                    ori_shape=img_shape,
                    scale_factor=scale_factor,
                    pad_param=pad_param,
                    text=grounding_prompt
                    )
    data_sample = DetDataSample(metainfo=img_meta)

    batch_size = img_tensor.shape[0]

    input_img_255 = img_tensor255
    data_feed = dict(
        inputs=[x for x in input_img_255],
        data_samples=[data_sample]*batch_size
    )
    return data_feed







def get_adv_loss(model, resize_transform, pad_transform, image_bchw_rgb_01, 
                iou_thr,
                fg_mask=None, gt_bbox=None, device='cuda'):
    
    if fg_mask is not None and fg_mask.sum() == 0:
        return torch.tensor(0.0, device=device), data_feed
    
    data_feed = get_data_feed(image_bchw_rgb_01, resize_transform, pad_transform, grounding_prompt)

    data_feed = model.data_preprocessor(data_feed, training=False)


    preds = model.forward(inputs=data_feed['inputs'], data_samples=data_feed['data_samples'], mode='predict')
    scores = preds[0]._pred_instances.scores
    bboxes = preds[0]._pred_instances.bboxes

    
    bbox_pred = torch.cat([bboxes, scores[:, None]], dim=1)

    if fg_mask is None:
        gt_box = gt_bbox[None]
    else:
        gt_box = get_gt_bbox(fg_mask, device)[None]

    gt_box = gt_box.to(device).float()
    iou = bbox_overlaps(bbox_pred[:,:4], gt_box)
    iou_mask = (iou>iou_thr).sum(dim=1)
    all_bbox_score_loss = (iou_mask * bbox_pred[:,4]).sum()


    return all_bbox_score_loss, data_feed





def get_pred(model_plus, image_bchw_rgb_01, device='cuda'):

    model, resize_transform, pad_transform, inferencer = model_plus
    
    data_feed_batch = get_data_feed(image_bchw_rgb_01, resize_transform, pad_transform, grounding_prompt)

    data_feed_batch = model.data_preprocessor(data_feed_batch, training=False)
    preds_batch = model.forward(inputs=data_feed_batch['inputs'], data_samples=data_feed_batch['data_samples'], mode='predict')

    bbox_pred_list = []
    labels_list = []
    for preds in preds_batch:
        scores = preds._pred_instances.scores
        bboxes = preds._pred_instances.bboxes
        labels = preds._pred_instances.labels

        bbox_pred = torch.cat([bboxes, scores[:, None]], dim=1)

        bbox_pred_list.append(bbox_pred)
        labels_list.append(labels)
    return bbox_pred_list, labels_list



def get_pred_with_mask(model_plus, image_bchw_rgb_01, device='cuda'):

    model, resize_transform, pad_transform, inferencer = model_plus
    
    data_feed_batch = get_data_feed(image_bchw_rgb_01, resize_transform, pad_transform, grounding_prompt)


    data_feed_batch = model.data_preprocessor(data_feed_batch, training=False)
    preds_batch = model.forward(inputs=data_feed_batch['inputs'], data_samples=data_feed_batch['data_samples'], mode='predict')

    bbox_pred_list = []
    labels_list = []
    masks_list = []
    for preds in preds_batch:
        scores = preds._pred_instances.scores
        bboxes = preds._pred_instances.bboxes
        labels = preds._pred_instances.labels
        masks = preds._pred_instances.masks

        bbox_pred = torch.cat([bboxes, scores[:, None]], dim=1)

        bbox_pred_list.append(bbox_pred)
        labels_list.append(labels)
        masks_list.append(masks)
    return bbox_pred_list, labels_list, masks_list





def get_detected_result(model, resize_transform, pad_transform, image_bchw_rgb_01, 
                        iou_thr, score_thr,
                        fg_mask=None, gt_bbox=None, device='cuda'):
    
    if fg_mask is not None and fg_mask.sum() == 0:
        return 0, 0

    data_feed = get_data_feed(image_bchw_rgb_01, resize_transform, pad_transform, grounding_prompt)

    data_feed = model.data_preprocessor(data_feed, training=False) 
    with torch.no_grad():
        preds = model.forward(inputs=data_feed['inputs'], data_samples=data_feed['data_samples'], mode='predict')
    scores = preds[0]._pred_instances.scores
    bboxes = preds[0]._pred_instances.bboxes

    bbox_pred = torch.cat([bboxes, scores[:, None]], dim=1)

    
    # 找到 gt bbox
    if fg_mask is None:
        gt_box = gt_bbox[None]
    else:
        gt_box = get_gt_bbox(fg_mask, device)[None]

    iou = bbox_overlaps(bbox_pred[:,:4], gt_box)
    iou_mask = (iou>iou_thr).sum(dim=1)
    score_mask = (bbox_pred[:,4]>score_thr)
    valid_mask = iou_mask * score_mask

    if valid_mask.sum() == 0:
        return 0, 1
    else:
        return 1, 1


def save_detect_from_data_feed(model, image_bchw_rgb_01, data_feed, inferencer, result_save_path, show_score_thr=0.3):


    with torch.no_grad():
        preds = model.forward(inputs=data_feed['inputs'], data_samples=data_feed['data_samples'], mode='predict')

    preds[0]._pred_instances.bboxes = preds[0]._pred_instances.bboxes.detach()
    preds[0]._pred_instances.scores = preds[0]._pred_instances.scores.detach()

    visualizer = inferencer.visualizer

    img = (image_bchw_rgb_01[0]*255).detach().cpu().numpy().transpose(1,2,0).astype(np.uint8)


    img_name = osp.basename(result_save_path)

    pred_score_thr = show_score_thr

    show = False
    draw_pred = True
    wait_time = 0

    pred = preds[0]

    visualizer.add_datasample(
                img_name,
                img,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                out_file=result_save_path,
            )



def save_detect_from_image(model_plus, image_bchw_rgb_01, result_save_path, show_score_thr=0.3):

    model, resize_transform, pad_transform, inferencer = model_plus


    data_feed = get_data_feed(image_bchw_rgb_01, resize_transform, pad_transform, grounding_prompt)

    data_feed = model.data_preprocessor(data_feed, training=False)
    save_detect_from_data_feed(model, image_bchw_rgb_01, data_feed, inferencer, result_save_path, show_score_thr)


def save_detect_from_image_batch(model_plus, image_bchw_rgb_01, result_save_path, show_score_thr=0.3):

    batch_size = image_bchw_rgb_01.shape[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(batch_size):
            temp_save_path = osp.join(temp_dir, f'{i}.jpg')
            save_detect_from_image(model_plus, image_bchw_rgb_01[i:i+1], temp_save_path, show_score_thr)
        result_image_list = []
        for i in range(batch_size):
            temp_save_path = osp.join(temp_dir, f'{i}.jpg')
            image_i = open_img(temp_save_path)
            result_image_list.append(image_i)
        save_image(torch.cat(result_image_list, dim=2), result_save_path, padding=0)

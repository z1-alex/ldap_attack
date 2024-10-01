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

from extend.utils.homo_affine import homography_batch






def _render_belly_patch_wide_batch(image, belly_patch, belly_mask, a5s,a6s,a11s,a12s, v5_6_11_12s, width):
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
    ]).repeat(batch_size_now,1,1).to(image.device)


    c5_6s = ( a5s + a6s )/2
    c11_12s = ( a11s + a12s )/2




    c56_to_5 = a5s - c5_6s
    c56_to_5_norm = c56_to_5/torch.sqrt(c56_to_5[:,0].pow(2)+c56_to_5[:,1].pow(2))


    x1s = torch.stack([
        c5_6s + c56_to_5_norm * width/2 + (c11_12s - c5_6s)/4,    # left shoulder
        c5_6s - c56_to_5_norm * width/2 + (c11_12s - c5_6s)/4,    # right shoulder
        c5_6s - c56_to_5_norm * width/2 + (c11_12s - c5_6s)*3/4,  # right hip
        c5_6s + c56_to_5_norm * width/2 + (c11_12s - c5_6s)*3/4   # left hip
    ],dim=1).to(image.device)

    # judge by v5_6_11_12s
    # the False also takes affine, but from x2s to x2s
    # this is for preventing no solution in homography()
    x1s = torch.where(v5_6_11_12s.unsqueeze(-1).unsqueeze(-1), x1s, x2s)


    # theta_line = homography(x1s[0], x2s[0])

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

    # 先x后y
    # patch 初始位置
    x2s = torch.Tensor([
        [_x_end, _y_start],
        [_x_start, _y_start],
        [_x_start, _y_end],
        [_x_end, _y_end],
    ]).repeat(batch_size_now,1,1).to(image.device)




    direction_vector_batch = down_point - up_point

    # （左右翻转的）左上，右上，右下，左下
    #   2       1
    # 
    # 
    #   3       4
    
    # 求解与方向向量垂直的方向

    orthogonal_vector_batch = torch.where(
        (direction_vector_batch[:,1] == 0).unsqueeze(-1),
        torch.Tensor([0,1]).to(image.device).unsqueeze(0).repeat(batch_size_now,1),
        torch.stack([torch.Tensor([1]*batch_size_now).to(image.device), - (direction_vector_batch[:,0]/direction_vector_batch[:,1])], dim=-1))

    # orthogonal_vector_batch2 = []
    # for i in range(batch_size_now):
    #     if direction_vector_batch[i,1] == 0:
    #         orthogonal_vector = torch.Tensor([0,1])
    #     else:
    #         orthogonal_vector = torch.Tensor([1, - direction_vector_batch[i,0]/direction_vector_batch[i,1]])
    #     orthogonal_vector_batch2.append(orthogonal_vector)
    # orthogonal_vector_batch2 = torch.stack(orthogonal_vector_batch2, dim=0)
    
    orthogonal_vector_norm_batch = orthogonal_vector_batch/ torch.sqrt(orthogonal_vector_batch[:,0].pow(2)+orthogonal_vector_batch[:,1].pow(2)).unsqueeze(-1)
    # orthogonal_vector_with_width_batch = orthogonal_vector_norm_batch * torch.Tensor([width]).to(image.device).unsqueeze(-1) /2

    def cross_product(v1,v2):
        return v1[0]*v2[1]-v2[0]*v1[1]
    def cross_product_batch(v1,v2):
        return v1[:,0]*v2[:,1]-v2[:,0]*v1[:,1]

    # orthogonal_vector_with_width_batch = torch.where(
    #     (cross_product_batch(direction_vector_batch, orthogonal_vector_with_width_batch) < 0).unsqueeze(-1),
    #     -orthogonal_vector_with_width_batch,
    #     orthogonal_vector_with_width_batch
    # )

    
    # a1 = up_point - orthogonal_vector_with_width_batch
    # a2 = up_point + orthogonal_vector_with_width_batch
    # a3 = down_point + orthogonal_vector_with_width_batch
    # a4 = down_point - orthogonal_vector_with_width_batch

    # a1 = 3/4*up_point + 1/4*down_point - orthogonal_vector_with_width_batch
    # a2 = 3/4*up_point + 1/4*down_point + orthogonal_vector_with_width_batch
    # a3 = 1/4*up_point + 3/4*down_point + orthogonal_vector_with_width_batch
    # a4 = 1/4*up_point + 3/4*down_point - orthogonal_vector_with_width_batch


    # 重新计算面积分配高和宽，让最终呈现2:1的高宽比

    area_ = width * (up_point-down_point).pow(2).sum(1).sqrt() /2 
    width_new = (area_/2).sqrt()

    orthogonal_vector_with_width_batch = orthogonal_vector_norm_batch * torch.Tensor([width_new]).to(image.device).unsqueeze(-1) /2
    orthogonal_vector_with_width_batch = torch.where(
        (cross_product_batch(direction_vector_batch, orthogonal_vector_with_width_batch) < 0).unsqueeze(-1),
        -orthogonal_vector_with_width_batch,
        orthogonal_vector_with_width_batch
    )
    center_point_batch = 1/2*up_point + 1/2*down_point
    up_down_vector_batch = down_point - up_point
    up_down_vector_norm_batch = up_down_vector_batch / up_down_vector_batch.pow(2).sum(1).sqrt().unsqueeze(-1)
    up_down_vector_with_height_batch = up_down_vector_norm_batch * torch.Tensor([width_new]).to(image.device).unsqueeze(-1)


    a1 = center_point_batch - up_down_vector_with_height_batch - orthogonal_vector_with_width_batch
    a2 = center_point_batch - up_down_vector_with_height_batch + orthogonal_vector_with_width_batch
    a3 = center_point_batch + up_down_vector_with_height_batch + orthogonal_vector_with_width_batch
    a4 = center_point_batch + up_down_vector_with_height_batch - orthogonal_vector_with_width_batch

    # a1 = 3/4*up_point + 1/4*down_point - orthogonal_vector_with_width_batch
    # a2 = 3/4*up_point + 1/4*down_point + orthogonal_vector_with_width_batch
    # a3 = 1/4*up_point + 3/4*down_point + orthogonal_vector_with_width_batch
    # a4 = 1/4*up_point + 3/4*down_point - orthogonal_vector_with_width_batch







    # 左右肩中间点和左右臀的中间点的连线，作为延长向量
    # 从左右肩延长出来这个向量，构成一个平行四边形
    # 作为放置目标

    

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

    # theta = torch.Tensor([
    #     [b11, b12, b13],
    #     [b21, b22, b23]
    #     ])

    theta = torch.stack([
        torch.stack([b11, b12, b13],dim=-1),
        torch.stack([b21, b22, b23],dim=-1),
        ], dim=1)
    ###########################

    # theta_line = homography(x1s, x2s)

    # a11,a12,a13,a21,a22,a23,_,_ = list(theta_line)
    # w = image.shape[-1]
    # h = image.shape[-2]
    # b11 = a11
    # b12 = a12*h/w
    # b13 = a13/(w/2)+a11+h/w*a12-1
    # b21 = a21*w/h
    # b22 = a22
    # b23 = a21*w/h+a22+a23/(h/2)-1

    # theta = torch.Tensor([
    #     [b11, b12, b13],
    #     [b21, b22, b23]
    #     ])

    grid = F.affine_grid(theta, image.size(), align_corners=False)
    limb_patch_affine = F.grid_sample(limb_patch_pad, grid, align_corners=False).squeeze()
    limb_mask_affine = F.grid_sample(limb_mask_pad, grid, align_corners=False).squeeze()

    patched_ = torch.where(limb_mask_affine>0, limb_patch_affine, image)
    masked_ = torch.where(limb_mask_affine>0, limb_mask_affine, torch.zeros_like(limb_mask_affine))


    patched_varify = torch.where(value_flag.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), patched_, image)
    masked_varify = torch.where(value_flag.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), masked_, torch.zeros_like(masked_))

    return patched_varify, masked_varify
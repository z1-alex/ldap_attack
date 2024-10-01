import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F


def diff_resize(results, resize_transform):

    image = results['img']
    image_BHWC = image.permute(0, 2, 3, 1)
    results = dict(img=image_BHWC)
     
    if resize_transform.scale:
        results['scale'] = resize_transform.scale
    else:
        from mmcv.image.geometric import _scale_size
        img_shape = results['img'].shape[1:3]
        results['scale'] = _scale_size(img_shape[::-1], resize_transform.scale_factor)

    if results.get('img', None) is not None:
        if resize_transform.keep_ratio:

            # img, scale_factor = mmcv.imrescale(
            #     results['img'],
            #     results['scale'],
            #     interpolation=resize_transform.interpolation,
            #     return_scale=True,
            #     backend=resize_transform.backend)
            model_img_scale = resize_transform.scale
            target_width, target_height = model_img_scale

            orig_height, orig_width = results['img'].shape[1:3]
            
            scale_w = target_width / orig_width
            scale_h = target_height / orig_height
            
            scale = min(scale_w, scale_h)
            
            new_width, new_height = int(orig_width * scale), int(orig_height * scale)

            img = transforms.Resize((new_height, new_width), antialias=False)(image)

            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            # new_h, new_w = img.shape[:2]
            new_h, new_w = img.shape[-2:]
            h, w = results['img'].shape[1:3]
            w_scale = new_w / w
            h_scale = new_h / h
        else:


            new_height, new_width = results['scale']

            img = transforms.Resize((new_height, new_width))(image)

            h, w = results['img'].shape[1:3]
            w_scale = new_width / w
            h_scale = new_height / h


        results['img'] = img
        # results['img_shape'] = img.shape[:2]
        results['img_shape'] = tuple(img.shape[-2:])
        results['scale_factor'] = (w_scale, h_scale)
        results['keep_ratio'] = resize_transform.keep_ratio

        return results




def diff_pad(results, pad_transform):
    # mmcv/transforms/processing.py

    image = results['img']
    image_BHWC = image.permute(0, 2, 3, 1)
    results['img'] = image_BHWC


    pad_val = pad_transform.pad_val.get('img', 0)

    size = None
    if pad_transform.pad_to_square:
        max_size = max(results['img'].shape[1:3])
        size = (max_size, max_size)
    if pad_transform.size_divisor is not None:
        if size is None:
            size = (results['img'].shape[1], results['img'].shape[2])
        pad_h = int(np.ceil(
            size[0] / pad_transform.size_divisor)) * pad_transform.size_divisor
        pad_w = int(np.ceil(
            size[1] / pad_transform.size_divisor)) * pad_transform.size_divisor
        size = (pad_h, pad_w)
    elif pad_transform.size is not None:
        size = pad_transform.size[::-1]
    if isinstance(pad_val, int) and results['img'].ndim == 4:
        pad_val = tuple(pad_val for _ in range(results['img'].shape[3]))

    # padded_img = mmcv.impad(
    #     results['img'],
    #     shape=size,
    #     pad_val=pad_val,
    #     padding_mode=pad_transform.padding_mode)
        
    # torch  pad
    B, C, H, W = image.shape
    padded_img = torch.zeros(B, C, size[0], size[1], device=image.device, dtype=image.dtype)
    for i in range(C):
        padded_img[:, i] = pad_val[:, i]
    padded_img[:, :, :H, :W] = image


    results['img'] = padded_img
    # results['pad_shape'] = padded_img.shape
    results['pad_shape'] = tuple(padded_img.shape)
    results['pad_fixed_size'] = pad_transform.size
    results['pad_size_divisor'] = pad_transform.size_divisor
    # results['img_shape'] = padded_img.shape[:2]
    results['img_shape'] = tuple(padded_img.shape[-2:])

    return results




def diff_pad_letter(results, pad_transform):


    
    image_BCHW = results['img']
    image_BHWC = image_BCHW.permute(0, 2, 3, 1)
    image_shape = image_BHWC.shape[1:3]  # height, width
    scale = pad_transform.scale[::-1]  # wh -> hw

    # Scale ratio (new / old)
    ratio = min(scale[0] / image_shape[0], scale[1] / image_shape[1])

    # only scale down, do not scale up (for better test mAP)
    if not pad_transform.allow_scale_up:
        ratio = min(ratio, 1.0)

    ratio = [ratio, ratio]  # float -> (float, float) for (height, width)

    # compute the best size of the image
    no_pad_shape = (int(round(image_shape[0] * ratio[0])),
                    int(round(image_shape[1] * ratio[1])))

    # padding height & width
    padding_h, padding_w = [
        scale[0] - no_pad_shape[0], scale[1] - no_pad_shape[1]
    ]
    if pad_transform.use_mini_pad:
        # minimum rectangle padding
        padding_w, padding_h = np.mod(padding_w, 32), np.mod(padding_h, 32)

    elif pad_transform.stretch_only:
        # stretch to the specified size directly
        padding_h, padding_w = 0.0, 0.0
        no_pad_shape = (scale[0], scale[1])
        ratio = [scale[0] / image_shape[0],
                    scale[1] / image_shape[1]]  # height, width ratios

    if image_shape != no_pad_shape:

        image_BHWC = F.interpolate(
            image_BHWC.permute(0, 3, 1, 2),
            size=(no_pad_shape[0], no_pad_shape[1]),
            align_corners=False).permute(0, 2, 3, 1)

    scale_factor = (no_pad_shape[1] / image_shape[1],
                    no_pad_shape[0] / image_shape[0])

    if 'scale_factor' in results:
        results['scale_factor_origin'] = results['scale_factor']
    results['scale_factor'] = scale_factor

    # padding
    top_padding, left_padding = int(round(padding_h // 2 - 0.1)), int(
        round(padding_w // 2 - 0.1))
    bottom_padding = padding_h - top_padding
    right_padding = padding_w - left_padding

    padding_list = [
        top_padding, bottom_padding, left_padding, right_padding
    ]
    if top_padding != 0 or bottom_padding != 0 or \
            left_padding != 0 or right_padding != 0:

        pad_val = pad_transform.pad_val.get('img', 0)
        if isinstance(pad_val, int) and image_BHWC.ndim == 4:
            pad_val = tuple(pad_val for _ in range(image_BHWC.shape[3]))



        padded_img = F.pad(image_BHWC.permute(0, 3, 1, 2),
                           (padding_list[2], padding_list[3], padding_list[0],padding_list[1]), 
                           value=-1).permute(0, 2, 3, 1)

        padded_img_flatten = padded_img.view(-1, 3)
        pad_mask = padded_img_flatten[:,0] == -1
        padded_img_flatten[pad_mask, 0] = pad_val[0]
        padded_img_flatten[pad_mask, 1] = pad_val[1]
        padded_img_flatten[pad_mask, 2] = pad_val[2]
        padded_img = padded_img_flatten.view(padded_img.shape)
        image_BHWC = padded_img
 
    
    results['img'] = image_BHWC.permute(0, 3, 1, 2)
    results['img_shape'] = tuple(image_BHWC.shape[1:3])
    if 'pad_param' in results:
        results['pad_param_origin'] = results['pad_param'] * \
                                        np.repeat(ratio, 2)
    if pad_transform.half_pad_param:
        results['pad_param'] = np.array(
            [padding_h / 2, padding_h / 2, padding_w / 2, padding_w / 2],
            dtype=np.float32)
    else:
        # We found in object detection, using padding list with
        # int type can get higher mAP.
        results['pad_param'] = np.array(padding_list, dtype=np.float32)

    results['pad_shape'] = tuple(image_BHWC.shape[1:3])

    return results



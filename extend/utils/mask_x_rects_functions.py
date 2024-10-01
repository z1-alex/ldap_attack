import torch
from extend.utils.render_patch_use_keypoint_with_width \
    import _render_limb_patch_wide_batch,_render_belly_patch_wide_batch,\
         _render_limb_patch_1v2_batch, _render_belly_patch_1v2_batch




def mask_dpatch(
        mask_area, 
        img_shape,
        image_real_mask=None,
        ):
    '''
     [  ]-----------------------+
        |                       |
        |                       |
        |                       |
        +-----------------------+

    '''

    mask = torch.zeros(img_shape)

    # image

    if image_real_mask != None:
        image_real_mask_x_line = image_real_mask[0].sum(-2).nonzero()
        image_real_mask_y_line = image_real_mask[0].sum(-1).nonzero()
        image_w = image_real_mask_x_line.max() - image_real_mask_x_line.min()
        image_h = image_real_mask_y_line.max() - image_real_mask_y_line.min()
        square_area_rate = mask_area / (image_w*image_h)
        mask_x1 = image_real_mask_x_line.min()
        mask_y1 = image_real_mask_y_line.min()
    else:

        image_w = img_shape[-1]
        image_h = img_shape[-2]
        square_area_rate = mask_area / (image_w*image_h)
        mask_x1 = 0
        mask_y1 = 0

        # mask_xc = object_xc
        # mask_yc = object_yc
    
    mask_h = image_h*square_area_rate.sqrt()
    mask_w = image_w*square_area_rate.sqrt()
    mask_xc = mask_x1 + mask_w/2
    mask_yc = mask_y1 + mask_h/2

    mask = make_1_rect(mask, mask_xc, mask_yc, mask_w, mask_h)

    return mask



def mask_1_rect(
        mask_area, 
        object_x1, object_x2, 
        object_y1, object_y2,  
        img_shape
        ):
    '''
        +-----------------------+
        |                       |
        |         [  ]          |
        |                       |
        +-----------------------+

    '''
    object_w = (object_x1 - object_x2).abs()
    object_h = (object_y1 - object_y2).abs()
    object_xc = (object_x1 + object_x2) / 2
    object_yc = (object_y1 + object_y2) / 2

    square_area_rate = mask_area / (object_w*object_h)

    mask = torch.Tensor(img_shape).fill_(0)
    mask_h = object_h*square_area_rate.sqrt()
    mask_w = object_w*square_area_rate.sqrt()

    mask_xc = object_xc
    mask_yc = object_yc

    image_w = img_shape[-1]
    image_h = img_shape[-2]
    mask = make_1_rect(mask, mask_xc, mask_yc, mask_w, mask_h, image_w, image_h)

    return mask


def mask_2_rects_vertical(
        mask_area, 
        object_x1, object_x2, 
        object_y1, object_y2, 
        img_shape
    ):
    '''
        +-----------------------+
        |                       |
        |         [  ]          |
        |                       |
        +-----------------------+
        |                       |
        |         [  ]          |
        |                       |
        +-----------------------+

    '''

    object_xc = (object_x1 + object_x2) / 2
    object_yc = (object_y1 + object_y2) / 2
    object_w = (object_x1 - object_x2).abs()
    object_h = (object_y1 - object_y2).abs()

    square_area_rate = (mask_area /2) / (object_w*(object_h/2))

    mask = torch.Tensor(img_shape).fill_(0)
    mask_h = (object_h/2)*square_area_rate.sqrt()
    mask_w = object_w*square_area_rate.sqrt()

    image_w = img_shape[-1]
    image_h = img_shape[-2]

    # up patch
    mask_xc = object_xc
    mask_yc = (object_y1 + object_yc) / 2
    mask = make_1_rect(mask, mask_xc, mask_yc, mask_w, mask_h, image_w, image_h)

    # below patch
    mask_xc = object_xc
    mask_yc = (object_yc + object_y2) / 2
    mask = make_1_rect(mask, mask_xc, mask_yc, mask_w, mask_h, image_w, image_h)

    return mask

def mask_2_rects_horizontal(
        mask_area, 
        object_x1, object_x2, 
        object_y1, object_y2, 
        img_shape
    ):
    '''
        +-----------------------+-----------------------+
        |                       |                       |
        |         [  ]          |         [  ]          |
        |                       |                       |
        +-----------------------+-----------------------+

    '''

    object_xc = (object_x1 + object_x2) / 2
    object_yc = (object_y1 + object_y2) / 2
    object_w = (object_x1 - object_x2).abs()
    object_h = (object_y1 - object_y2).abs()

    square_area_rate = (mask_area /2) / ((object_w/2)*object_h)

    mask = torch.Tensor(img_shape).fill_(0)
    mask_h = object_h*square_area_rate.sqrt()
    mask_w = (object_w/2)*square_area_rate.sqrt()

    image_w = img_shape[-1]
    image_h = img_shape[-2]

    # up patch
    mask_xc = (object_x1 + object_xc) / 2
    mask_yc =  object_yc
    mask = make_1_rect(mask, mask_xc, mask_yc, mask_w, mask_h, image_w, image_h)

    # below patch
    mask_xc = (object_xc + object_x2) / 2
    mask_yc =  object_yc
    mask = make_1_rect(mask, mask_xc, mask_yc, mask_w, mask_h, image_w, image_h)

    return mask


def mask_4_rects(
        mask_area, 
        object_x1, object_x2, 
        object_y1, object_y2, 
        img_shape
    ):
    '''
        +-----------------------+-----------------------+
        |                       |                       |
        |         [  ]          |         [  ]          |
        |                       |                       |
        +-----------------------+-----------------------+
        |                       |                       |
        |         [  ]          |         [  ]          |
        |                       |                       |
        +-----------------------+-----------------------+
        
    '''
    object_xc = (object_x1 + object_x2) / 2
    object_yc = (object_y1 + object_y2) / 2
    object_w = (object_x1 - object_x2).abs()
    object_h = (object_y1 - object_y2).abs()

    square_area_rate = mask_area /4 / (object_w*object_h)

    mask = torch.Tensor(img_shape).fill_(0)
    mask_h = object_h*square_area_rate.sqrt()
    mask_w = object_w*square_area_rate.sqrt()

    image_w = img_shape[-1]
    image_h = img_shape[-2]

    mask_xc = (object_x1 + object_xc) / 2
    mask_yc = (object_y1 + object_yc) / 2
    mask = make_1_rect(mask, mask_xc, mask_yc, mask_w, mask_h, image_w, image_h)

    mask_xc = (object_x2 + object_xc) / 2
    mask_yc = (object_y1 + object_yc) / 2
    mask = make_1_rect(mask, mask_xc, mask_yc, mask_w, mask_h, image_w, image_h)

    mask_xc = (object_x1 + object_xc) / 2
    mask_yc = (object_y2 + object_yc) / 2
    mask = make_1_rect(mask, mask_xc, mask_yc, mask_w, mask_h, image_w, image_h)

    mask_xc = (object_x2 + object_xc) / 2
    mask_yc = (object_y2 + object_yc) / 2
    mask = make_1_rect(mask, mask_xc, mask_yc, mask_w, mask_h, image_w, image_h)
    return mask




def mask_10_rects(
        mask_area, 
        object_x1, object_x2, 
        object_y1, object_y2, 
        img_shape
    ):
    '''
        +-----------------------+-----------------------+
        |                       |                       |
        |         [  ]          |         [  ]          |
        |                       |                       |
        +-----------------------+-----------------------+
        |                       |                       |
        |         [  ]          |         [  ]          |
        |                       |                       |
        +-----------------------+-----------------------+
        
    '''
    object_xc = (object_x1 + object_x2) / 2
    object_yc = (object_y1 + object_y2) / 2
    object_w = (object_x1 - object_x2).abs()
    object_h = (object_y1 - object_y2).abs()

    mask_number = 10
    x1y1x2y2 = torch.Tensor([object_x1,object_y1,object_x2,object_y2])
    mask_c_xy = unify_distribution_xy_by_box(mask_number, x1y1x2y2, img_shape[-2:])



    square_area_rate = mask_area /mask_number / (object_w*object_h)

    mask = torch.Tensor(img_shape).fill_(0)
    mask_h = object_h*square_area_rate.sqrt()
    mask_w = object_w*square_area_rate.sqrt()

    image_w = img_shape[-1]
    image_h = img_shape[-2]


    for i in range(mask_c_xy.shape[0]):
        mask_xc = mask_c_xy[i, 0] * img_shape[-1]
        mask_yc = mask_c_xy[i, 1] * img_shape[-2]
        mask = make_1_rect(mask, mask_xc, mask_yc, mask_w, mask_h, image_w, image_h)



    return mask



def unify_distribution_xy_by_box(num, x1y1x2y2, img_size):
    num = torch.Tensor([num])
    base_num_a = int(torch.sqrt(num))
    if base_num_a*base_num_a <= num < base_num_a*(base_num_a + 1):
        base_num_b = base_num_a 
    elif base_num_a*(base_num_a + 1) <= num < (base_num_a+1)*(base_num_a + 1):
        base_num_b = base_num_a + 1

    x1 = x1y1x2y2[0]/ img_size[0]
    y1 = x1y1x2y2[1]/ img_size[1]
    x2 = x1y1x2y2[2]/ img_size[0]
    y2 = x1y1x2y2[3]/ img_size[1]
    
    x_a_split = torch.linspace(x1,x2,int(base_num_a*2+1))
    y_b_split = torch.linspace(y1,y2,int(base_num_b*2+1))

    x_split = x_a_split[[2*i+1 for i in range(base_num_a)]]
    y_split = y_b_split[[2*j+1 for j in range(base_num_b)]]
    
    xy = torch.stack([
        x1 + torch.rand(int(num)).to(x1.device)*(x2-x1),
        y1 + torch.rand(int(num)).to(y1.device)*(y2-y1)
        ], dim=1)
    for i in range(base_num_a):
        for j in range(base_num_b):
            xy[i*base_num_a + j] = torch.Tensor([x_split[i], y_split[j]])
    return xy

def make_1_rect(mask, mask_xc, mask_yc, mask_w, mask_h, image_w, image_h):
    mask_x_start = int(mask_xc - mask_w/2)
    mask_x_end = int(mask_x_start + mask_w)
    mask_y_start = int(mask_yc - mask_h/2)
    mask_y_end = int(mask_y_start + mask_h)

    mask_x_start = min(max(mask_x_start, 0), image_w-1)
    mask_x_end = min(max(mask_x_end, 0), image_w)
    mask_y_start = min(max(mask_y_start, 0), image_h-1)
    mask_y_end = min(max(mask_y_end, 0), image_h)


    mask[:,mask_y_start:mask_y_end,mask_x_start:mask_x_end] = 1
    return mask



def body_parts_7(
        mask_area, 
        keypoints,
        img_shape
    ):

    # 7 body parts! 7BP!

    need_imgs_limb = [
                'left_up_arm',
                # 'left_down_arm',

                'right_up_arm',
                # 'right_down_arm',

                'left_up_leg',
                'left_down_leg',

                'right_up_leg',
                'right_down_leg',
            
                
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
    


    device = keypoints.device
    mask_area_expect = mask_area.float()

    mask = torch.Tensor(img_shape).fill_(0).to(device)
    kp = keypoints

    # keypoints
    x = kp[0::3].float()
    y = kp[1::3].float()
    v = kp[2::3]

    
    init_unclear_len =  0 

    for key in need_imgs_limb:
        x_d = x[skeletons_num_dict[key]][0] - x[skeletons_num_dict[key]][1]
        y_d = y[skeletons_num_dict[key]][0] - y[skeletons_num_dict[key]][1]
        init_unclear_len += torch.sqrt(x_d.pow(2)+y_d.pow(2))
    x_d = (x[5] + x[6])/2 - (x[11] + x[12])/2
    y_d = (y[5] + y[6])/2 - (y[11] + y[12])/2
    init_unclear_len += torch.sqrt(x_d.pow(2)+y_d.pow(2)) * 2


    unit_width = mask_area_expect / init_unclear_len * 2

    # patches ready to train

    common_fake_img = torch.Tensor(1,3,500,500).fill_(1).to(device)
    common_fake_patch = common_fake_img
    common_fake_mask = common_fake_img

    masked_mask = 0


    # if patch_dict['belly'] != None:
    a5s = torch.stack([x[5], y[5]]).unsqueeze(0)
    a6s = torch.stack([x[6], y[6]]).unsqueeze(0)
    a11s = torch.stack([x[11], y[11]]).unsqueeze(0)
    a12s = torch.stack([x[12], y[12]]).unsqueeze(0)
    v5_6_11_12s = torch.all(v[[5,6,11,12]]>0).unsqueeze(0)


    patched_image, _mask = _render_belly_patch_1v2_batch(
        common_fake_img,    # image
        common_fake_patch,    # belly_patch
        common_fake_mask,    # belly_mask
        a5s,a6s,a11s,a12s, 
        v5_6_11_12s,
        width=unit_width*2)

    masked_mask += _mask

    # 两点都存在，即可添加四肢 patch


    # use a for circle to write this 

    for key in need_imgs_limb:

        if 'arm' in key:
            width = unit_width
            width_key_num1 = 5
            width_key_num2 = 6
        elif 'leg' in key:
            width = unit_width
            width_key_num1 = 11
            width_key_num2 = 12
        else:
            print('no leg or arm in name! plz check')
            raise 

        up_point_num = skeletons_num_dict[key][0]
        down_point_num = skeletons_num_dict[key][1]

        part_patch = common_fake_patch
        part_mask = common_fake_mask
        up_point = torch.stack([x[up_point_num], y[up_point_num]]).unsqueeze(0)
        down_point = torch.stack([x[down_point_num], y[down_point_num]]).unsqueeze(0)
        value_flag = torch.all(v[[width_key_num1,width_key_num2,up_point_num,down_point_num]]>0).unsqueeze(0)
        _, _mask = _render_limb_patch_1v2_batch(
            common_fake_img, 
            part_patch, 
            part_mask, 
            up_point, 
            down_point, 
            width, 
            value_flag) 
        masked_mask += _mask


    mask = torch.where(masked_mask>0, torch.ones_like(masked_mask), torch.zeros_like(masked_mask))

    return mask


def main():
    pass

if __name__=='__main__':
    main()
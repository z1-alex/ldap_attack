
import math
import torch
import torchvision

# from extend.block_function import block_function_adp, block_function_adp_wide, block_variant_function_adp

def block_function_adp(block_center, block_wh, block_alpha, img_size=500):

    pi = math.pi
    assert block_center.shape[0] == block_wh.shape[0] 
    block_num = block_alpha.shape[0]

    f = torch.cuda.FloatTensor(img_size,img_size).fill_(0)

    base = torch.cuda.FloatTensor([i for i in range(img_size)])
    base = torch.cuda.FloatTensor([i for i in range(img_size)])
    y = base.unsqueeze(1).repeat(1, base.shape[0])
    x = base.unsqueeze(0).repeat(base.shape[0], 1)

    for i in range(block_num):

        block_center_x_now = block_center[i,0]*img_size
        block_center_y_now = block_center[i,1]*img_size
        block_w_now = block_wh[i,0]*img_size
        block_h_now = block_wh[i,1]*img_size

        cosine_active_mask = torch.cuda.FloatTensor(img_size,img_size).fill_(0)

        mask_start_y = torch.clamp(block_center_y_now-block_h_now/2, 0, img_size).int()
        mask_end_y = torch.clamp(block_center_y_now+block_h_now/2, 0, img_size).int()

        mask_start_x = torch.clamp(block_center_x_now-block_w_now/2, 0, img_size).int()
        mask_end_x = torch.clamp(block_center_x_now+block_w_now/2, 0, img_size).int()

        cosine_active_mask[
            mask_start_y:mask_end_y,
            mask_start_x:mask_end_x] = 1



        f_now_x = (torch.cos((x-block_center_x_now)/block_w_now*2*pi) + 1)/2
        f_now_y = (torch.cos((y-block_center_y_now)/block_h_now*2*pi) + 1)/2

        f_now_raw = f_now_x * f_now_y

        f_now_masked_weighted = cosine_active_mask * f_now_raw * block_alpha[i]

        # save(f_now_masked_weighted/f_now_masked_weighted.max(), '1.png')

        f += f_now_masked_weighted
    return f


def block_function_adp_wide(block_center, block_wh, block_alpha, img_size=500):
    pi = math.pi
    assert block_center.shape[0] == block_wh.shape[0] 
    block_num = block_alpha.shape[0]

    f = torch.cuda.FloatTensor(img_size,img_size).fill_(0)

    base = torch.cuda.FloatTensor([i for i in range(img_size)])
    base = torch.cuda.FloatTensor([i for i in range(img_size)])
    y = base.unsqueeze(1).repeat(1, base.shape[0])
    x = base.unsqueeze(0).repeat(base.shape[0], 1)

    for i in range(block_num):

        block_center_x_now = block_center[i,0]*img_size
        block_center_y_now = block_center[i,1]*img_size
        block_w_now = block_wh[i,0]*img_size
        block_h_now = block_wh[i,1]*img_size

        if block_w_now == 0 or block_h_now == 0:
            continue

        cosine_active_mask = torch.cuda.FloatTensor(img_size,img_size).fill_(0)

        mask_start_y = torch.clamp(block_center_y_now-block_h_now/2*1.5, 0, img_size).int()
        mask_end_y = torch.clamp(block_center_y_now+block_h_now/2*1.5, 0, img_size).int()

        mask_start_x = torch.clamp(block_center_x_now-block_w_now/2*1.5, 0, img_size).int()
        mask_end_x = torch.clamp(block_center_x_now+block_w_now/2*1.5, 0, img_size).int()

        cosine_active_mask[
            mask_start_y:mask_end_y,
            mask_start_x:mask_end_x] = 1

        # f_now_x = (torch.cos((x-block_center_x_now)/block_w_now*2*pi) + 1)/2
        f_now_x = cos_wall_1wid((x-block_center_x_now)/block_w_now*2*pi)
        # f_now_y = (torch.cos((y-block_center_y_now)/block_h_now*2*pi) + 1)/2
        f_now_y = cos_wall_1wid((y-block_center_y_now)/block_h_now*2*pi)

        f_now_raw = f_now_x * f_now_y

        f_now_masked_weighted = cosine_active_mask * f_now_raw * block_alpha[i]

        # save(f_now_masked_weighted/f_now_masked_weighted.max(), '1.png')

        f += f_now_masked_weighted
    return f



def cos_wall_1wid(x):
    pi = math.pi
    a_part = x < -pi/2
    mid_part =  (-pi/2 <= x) * (x <= pi/2)
    b_part = pi/2 < x
    result_x = a_part * ( torch.cos(x+pi/2) + 1 ) /2 \
        + mid_part * torch.ones_like(x) \
        + b_part * ( torch.cos(x-pi/2) + 1 ) /2
    return result_x


def block_variant_function_adp(block_mean, block_std, block_rou, block_lambda, img_size=500):
    '''
        # f(x,y) = 1/((2*pi*d1*d2*torch.sqrt(1-rou*rou)))*torch.exp(
            #     -1/(2*(1-rou*rou))*(
            #         (x-u1)*(x-u1)/d1/d1
            #         -2*rou*(x-u1)*(y-u2)/d1/d2
            #         +(y-u2)*(y-u2)/d2/d2
            #         )
            #     )
    '''
    import math
    pi = math.pi
    assert block_mean.shape[0] == block_std.shape[0] 
    block_num = block_mean.shape[0]

    f = torch.zeros(img_size,img_size).cuda()

     

    x = torch.linspace(0,img_size-1,img_size,device='cuda')
    y = x.clone().detach()
    x = x.unsqueeze(1).repeat(1, y.shape[0])
    y = y.unsqueeze(0).repeat(x.shape[0], 1)

    for i in range(block_num):
        block_mean_now = block_mean[i]
        block_std_now = block_std[i]
        block_rou_now = block_rou[i]
        u1 = block_mean_now[0]*img_size
        u2 = block_mean_now[1]*img_size
        d1 = block_std_now[0]*img_size
        d2 = block_std_now[1]*img_size
        rou = torch.tanh(block_rou_now)
        # rou = torch.Tensor([0])
        f_now = 1/((2*pi*d1*d2*torch.sqrt(1-rou*rou)))*torch.exp(
                -1/(2*(1-rou*rou))*(
                    (x-u1)*(x-u1)/d1/d1
                    -2*rou*(x-u1)*(y-u2)/d1/d2
                    +(y-u2)*(y-u2)/d2/d2
                    )
                )
        if f_now.max() < 1e-10:
            f_now = torch.zeros_like(f_now)
        else:
            f_now = f_now/f_now.max()
        f_now = f_now * block_lambda[i]
        # f = torch.max(f,f_now)
        f += f_now
    return f



def block_function(x,y,u1,u2,d1,d2,rou):
    '''
        # f(x,y) = 1/((2*pi*d1*d2*torch.sqrt(1-rou*rou)))*torch.exp(
            #     -1/(2*(1-rou*rou))*(
            #         (x-u1)*(x-u1)/d1/d1
            #         -2*rou*(x-u1)*(y-u2)/d1/d2
            #         +(y-u2)*(y-u2)/d2/d2
            #         )
            #     )
    '''
    import math
    pi = math.pi
    f = torch.zeros(x.shape[0], y.shape[0])
    x = x.unsqueeze(1).repeat(1, y.shape[0])
    y = y.unsqueeze(0).repeat(x.shape[0], 1)
    f = 1/((2*pi*d1*d2*torch.sqrt(1-rou*rou)))*torch.exp(
                -1/(2*(1-rou*rou))*(
                    (x-u1)*(x-u1)/d1/d1
                    -2*rou*(x-u1)*(y-u2)/d1/d2
                    +(y-u2)*(y-u2)/d2/d2
                    )
                
                )
    return f




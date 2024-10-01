import torch
from attackmethod import AttackMethod
from torchvision.utils import save_image
import torch.nn.functional as F
from extend.utils.image_io import open_img

from extend.utils.mask_x_rects_functions import unify_distribution_xy_by_box
from extend.utils.block_function import block_function_adp_wide
import torch.optim as optim


class LDAPatch(AttackMethod):
    def __init__(self, image, bbox, device):
        self.image = image
        img_shape = image.shape[-2:]
        bbox_x1y1x2y2 = bbox.float().cuda()
        self.img_shape = img_shape


        self.block_num = 10
        block_center = unify_distribution_xy_by_box(self.block_num, bbox_x1y1x2y2, img_shape).cuda()
        block_wh_log = torch.ones(self.block_num, 2, device=device).fill_(0.33)
        self.block_center = block_center
        self.block_wh_log = block_wh_log

        # start at random
        adv_patch = torch.rand_like(image)
        self.adv_patch = adv_patch

        self.train_step = 2000

        block_alpha = torch.ones(self.block_num,1, device=device).fill_(1e-7)
        self.block_alpha = block_alpha
        self.wh_loss_lambda_start = None # wh_loss_alpha_start
        self.wh_loss_lambda_end = None
        self.alpha_increase_step = 600
        self.wh_loss_lambda = 10

        self.wh_loss_lambda_init = 1e-2
        self.wh_loss_lambda_change_start_time = 200
        self.wh_loss_lambda_increase_value = 1e-4

        self.texturel2_loss_lambda_init = 1e-2
        self.texturel2_loss_lambda_change_start_time = 200
        self.texturel2_loss_lambda_increase_value = 1e-4

        self.best_adv_image = image.clone()
        self.best_mask = torch.ones_like(image[0])
        self.best_mask_region = img_shape[0] * img_shape[1]



    def get_use_patch_area(self, ):
        if len(self.mask.shape) == 2:
            area = (self.mask>0).sum()
        elif len(self.mask.shape) == 3:
            area = (self.mask[0]>0).sum()
        elif len(self.mask.shape) == 4:
            area = (self.mask[0,0]>0).sum()
        return area
    
    def get_mask(self, block_alpha_abs):
        block_wh = F.softplus(self.block_wh_log * 10) / 10
        block_wh_abs = block_wh.abs()
        block_attack_area = block_function_adp_wide(self.block_center, block_wh_abs, block_alpha_abs, img_size=self.img_shape[0])
        block_attack_area_norm = block_attack_area 
        block_attack_area_norm_clamp = block_attack_area_norm.abs()
        return block_attack_area_norm_clamp

    def get_texture(self, ):
        return self.adv_patch

    def get_adv_image(self, step):

        block_alpha_abs = self.block_alpha.abs().clamp(0, 1)
        
        mask = self.get_mask(block_alpha_abs)
        self.mask = mask
        patched_img = self.image * (1 - mask) + self.get_texture() * mask
        return patched_img


    def optimizable_params(self, ):
        return None


    def init_optimizer(self, ):

        block_center = self.block_center
        block_wh_log = self.block_wh_log
        block_alpha = self.block_alpha
        adv_patch = self.adv_patch

        block_center.requires_grad_(True)
        block_wh_log.requires_grad_(True)
        block_alpha.requires_grad_(True)
        adv_patch.requires_grad_(True)

        
        optimizer_t = optim.Adam([
            {'params': adv_patch, 'lr': 0.03},
        ], amsgrad=True)
        optimizer_x_s = optim.Adam([
            {'params': block_center, 'lr':0.01},
            {'params': block_wh_log, 'lr':0.01},
        ], amsgrad=True)
        optimizer_alpha = optim.Adam([
            {'params': block_alpha, 'lr':0.01},
        ], amsgrad=True)

        self.optimizer_t = optimizer_t
        self.optimizer_x_s = optimizer_x_s
        self.optimizer_alpha = optimizer_alpha

        self.optimizer_list = [optimizer_t, optimizer_x_s, optimizer_alpha]


       
    def optimize_step(self, step):
    
        optimizer = self.optimizer_list[step % 3]

        optimizer.step()
        optimizer.zero_grad()

        self.adv_patch.data.clamp_(0, 1)
                    


    def get_extra_loss(self, step):


        block_wh = F.softplus(self.block_wh_log * 10) * 3
        block_wh_abs = block_wh.abs() #.clamp(0.0001, 1)

        if self.wh_loss_lambda_start != None:
            if self.wh_loss_lambda_start == self.wh_loss_lambda_end:
                wh_loss_lambda_now = self.wh_loss_lambda_start
            elif step < self.lambda_change_start_time:
                wh_loss_lambda_now = self.wh_loss_lambda_start
            elif self.lambda_change_start_time <= step < self.lambda_change_end_time:
                wh_loss_lambda_now = \
                    (self.wh_loss_lambda_end - self.wh_loss_lambda_start) \
                    / (self.lambda_change_end_time - self.lambda_change_start_time)\
                    *(step - self.lambda_change_start_time) + self.wh_loss_lambda_start
            elif self.lambda_change_end_time <= step:
                wh_loss_lambda_now = self.wh_loss_lambda_end
        else:
            wh_loss_lambda_now = self.wh_loss_lambda
            
        self.wh_loss_lambda_init = 1e-2
        self.wh_loss_lambda_change_start_time = 200
        self.wh_loss_lambda_increase_value = 1e-4

        if step > self.wh_loss_lambda_change_start_time:
            wh_loss_lambda_now = self.wh_loss_lambda_init + (step - self.wh_loss_lambda_change_start_time) * self.wh_loss_lambda_increase_value
        else:
            wh_loss_lambda_now = self.wh_loss_lambda_init
        wh_loss_lambda_now = min(max(wh_loss_lambda_now, 0), 1)
        
        if step > self.texturel2_loss_lambda_change_start_time:
            texturel2_loss_lambda_now = self.texturel2_loss_lambda_init + (step - self.texturel2_loss_lambda_change_start_time) * self.texturel2_loss_lambda_increase_value
        else:
            texturel2_loss_lambda_now = self.texturel2_loss_lambda_init
        texturel2_loss_lambda_now = min(max(texturel2_loss_lambda_now, 0), 1)

        
        wh_loss = torch.mean(block_wh_abs[:,0]*block_wh_abs[:,1]) * wh_loss_lambda_now
        texture_l2_loss = (self.get_adv_image(step) - self.image).pow(2).mean() * texturel2_loss_lambda_now

        return wh_loss + texture_l2_loss
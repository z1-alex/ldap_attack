import torch
import torchvision.transforms as transforms
from skimage.segmentation import slic
import PIL.Image as Image

import torch.nn.functional as F


def main():

    # A * x = B
    A = torch.Tensor([
        [ 2, 2, 0, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 0, 1, 1 ],
    ])

    B = torch.Tensor([
        [ 4 ],
        [ 8 ],
        [ 10 ],
    ])

    X, LU = torch.solve(B, A)


    print()

    for i in range(1):
        adv_patch = Image.open('pytorch-YOLOv4/predictions3.png').convert('RGB')
        adv_patch = transforms.ToTensor()(adv_patch)
        x1s = torch.Tensor([
            [ 0, 0 ],
            [ 0, 500 ],
            [ 500, 0 ],
            [ 500, 500 ]
        ])

        # x2s = x1s + torch.randn_like(x1s)*0.1 * x1s.max()
        # x2s = torch.Tensor([
        #     [ 0, 0 ],
        #     [ 0, 400 ],
        #     [ 400, 0 ],
        #     [ 400, 400 ]
        # ])
        x2s = torch.Tensor([
            [ 20, 20 ],
            [ 20, 350 ],
            [ 350, 20 ],
            [ 350, 350 ]
        ])

        x1s = x1s - 250
        x2s = x2s - 250

        x1s[:,0] = x1s[:,0]/adv_patch.shape[2]
        x1s[:,1] = x1s[:,1]/adv_patch.shape[1]
        x2s[:,0] = x2s[:,0]/adv_patch.shape[2]
        x2s[:,1] = x2s[:,1]/adv_patch.shape[1]

        
        theta_line = homography(x2s, x1s)



        # show(adv_patch)

        ######################## affine ########################
        ######### part by part establish the affine grid #######(- x_center + 250)/250
        ########################################################

        theta = torch.Tensor([
            [theta_line[0],theta_line[1],theta_line[2]*((theta_line[0].pow(2)+theta_line[1].pow(2))).sqrt(),],
            [theta_line[3],theta_line[4],theta_line[5]*((theta_line[3].pow(2)+theta_line[4].pow(2))).sqrt(),],
        ])

        grid = F.affine_grid(theta.unsqueeze(0), adv_patch.unsqueeze(0).size())


        grid = F.affine_grid(theta.unsqueeze(0), adv_patch.unsqueeze(0).size())
        affine_patch = F.grid_sample(adv_patch.unsqueeze(0), grid).squeeze()

        x_all = affine_patch.sum(0).sum(0).nonzero()
        y_all = affine_patch.sum(0).sum(1).nonzero()
        print('x:',x_all.min(),x_all.max(),'y:',y_all.min(), y_all.max())
        print()
    print()


# affine
def homography(x1s, x2s):
    '''
        x1s, x2s: size:[4,2]
        x1s: 0,0  0,600  600,0 600,600
    '''

    def ax(p, q):
        return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q):
        return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    p = []
    p.append(ax(x1s[0], x2s[0]))
    p.append(ay(x1s[0], x2s[0]))

    p.append(ax(x1s[1], x2s[1]))
    p.append(ay(x1s[1], x2s[1]))

    p.append(ax(x1s[2], x2s[2]))
    p.append(ay(x1s[2], x2s[2]))

    p.append(ax(x1s[3], x2s[3]))
    p.append(ay(x1s[3], x2s[3]))

    # A is 8x8
    A = torch.Tensor(p)

    m = [[x2s[0][0], x2s[0][1], x2s[1][0], x2s[1][1], x2s[2][0], x2s[2][1], x2s[3][0], x2s[3][1]]]

    # P is 8x1
    P = torch.Tensor(m).t()

    # here we solve the linear system
    # we transpose the result for convenience
    return torch.solve(P,A)[0]
    # return tf.transpose(tf.matrix_solve_ls(A, P, fast=True))


# affine
def homography_batch(x1s, x2s):
    '''
        x1s, x2s: size:[b,4,2]
        x1s: 0,0  0,600  600,0 600,600
    '''

    def ax(p, q):
        ones = torch.ones_like(p[:,0])
        zeros = torch.zeros_like(p[:,0])
        return [p[:,0], p[:,1], ones, zeros, zeros, zeros, -p[:,0] * q[:,0], -p[:,1] * q[:,0]]

    def ay(p, q):
        ones = torch.ones_like(p[:,0])
        zeros = torch.zeros_like(p[:,0])
        return [zeros, zeros, zeros, p[:,0], p[:,1], ones, -p[:,0] * q[:,1], -p[:,1] * q[:,1]]

    p = []
    p.append(torch.stack(ax(x1s[:,0], x2s[:,0]), dim=1))
    p.append(torch.stack(ay(x1s[:,0], x2s[:,0]), dim=1))

    p.append(torch.stack(ax(x1s[:,1], x2s[:,1]), dim=1))
    p.append(torch.stack(ay(x1s[:,1], x2s[:,1]), dim=1))

    p.append(torch.stack(ax(x1s[:,2], x2s[:,2]), dim=1))
    p.append(torch.stack(ay(x1s[:,2], x2s[:,2]), dim=1))

    p.append(torch.stack(ax(x1s[:,3], x2s[:,3]), dim=1))
    p.append(torch.stack(ay(x1s[:,3], x2s[:,3]), dim=1))

    # A is 8x8
    A = torch.stack(p,dim=1)

    m = [x2s[:,0,0], x2s[:,0,1], x2s[:,1,0], x2s[:,1,1], x2s[:,2,0], x2s[:,2,1], x2s[:,3,0], x2s[:,3,1]]

    # P is 8x1
    P = torch.stack(m, dim=1).unsqueeze(-1)

    # here we solve the linear system

    return torch.solve(P,A)[0]

if __name__ == '__main__':
    main()

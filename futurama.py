#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, load_checkpoint

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images, save_images

from setting_struct import setting_option
from copy_utils import copy_function,get_pair

device = "cuda" if torch.cuda.is_available() else "cpu"


def real_gmm(opt, test_loader, model):
    model.to(device)
    model.eval()

    base_name = os.path.basename(opt.checkpoint)

    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)

    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)

    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        c_names = inputs['c_name']
        im_names = inputs['im_name']
        im = inputs['image'].to(device)
        im_pose = inputs['pose_image'].to(device)
        im_h = inputs['head'].to(device)
        shape = inputs['shape'].to(device)
        agnostic = inputs['agnostic'].to(device)
        c = inputs['cloth'].to(device)
        cm = inputs['cloth_mask'].to(device)
        im_c = inputs['parse_cloth'].to(device)
        im_g = inputs['grid_image'].to(device)

        grid, theta = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border', align_corners=False)
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros', align_corners=False)
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros', align_corners=False)

        # visuals = [[im_h, shape, im_pose],
        #            [c, warped_cloth, im_c],
        #            [warped_grid, (warped_cloth + im) * 0.5, im]]

        save_images(warped_cloth, im_names, c_names, warp_cloth_dir)
        save_images(warped_mask * 2 - 1, im_names, c_names, warp_mask_dir)

        t = time.time() - iter_start_time
        print('used time: %.3f' % t, flush=True)



def real_tom(opt, test_loader, model):
    model.to(device)
    model.eval()

    base_name = os.path.basename(opt.checkpoint)

    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)


    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        c_names = inputs['c_name']
        im_names = inputs['im_name']
        im = inputs['image'].to(device)
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']
        agnostic = inputs['agnostic'].to(device)
        c = inputs['cloth'].to(device)
        cm = inputs['cloth_mask'].to(device)

        outputs = model(torch.cat([agnostic, c], 1))
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        # visuals = [[im_h, shape, im_pose],
        #            [c, 2 * cm - 1, m_composite],
        #            [p_rendered, p_tryon, im]]

        save_images(p_tryon, im_names, c_names, try_on_dir)


        t = time.time() - iter_start_time
        print('used time: %.3f' % t, flush=True)


def main_real_gmm():
    # opt = get_opt()

    opt = setting_option(
        name="gmm_real",
        stage="GMM",
        datamode="real",
        data_list="real_pairs.txt",
        checkpoint="checkpoints/gmm_train_new/gmm_final.pth",
        save_count=100,
        shuffle=False
    )

    # create dataset
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    # if not os.path.exists(opt.tensorboard_dir):
    #     os.makedirs(opt.tensorboard_dir)
    # board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

    # create model & train
    if opt.stage == 'GMM':
        model = GMM(opt)

        # checkpoint_path = "checkpoints/gmm_train_new/gmm_final.pth"
        load_checkpoint(model, opt.checkpoint)

        with torch.no_grad():
            real_gmm(opt, train_loader, model)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)



def main_real_tom():
    # opt = get_opt()

    opt = setting_option(
        name="tom_real_new",
        stage="TOM",
        datamode="real",
        data_list="real_pairs.txt",
        checkpoint="checkpoints/tom_train_new/tom_final.pth",
        save_count=100,
        shuffle=False
    )

    # create dataset
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # # visualization
    # if not os.path.exists(opt.tensorboard_dir):
    #     os.makedirs(opt.tensorboard_dir)
    # board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))


    if opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)

        load_checkpoint(model, opt.checkpoint)

        with torch.no_grad():
            real_tom(opt, train_loader, model)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)






def futurama():
    
    main_real_gmm()
    
    src1 = r'./result/gmm_final.pth/real/warp-cloth/'
    tar1 = r'./data/real/warp-cloth/'
    src2 = r'./result/gmm_final.pth/real/warp-mask/'
    tar2 = r'./data/real/warp-mask/'
    
    copy_function(src1,tar1)
    copy_function(src2,tar2)

    note = r"./data/real_pairs.txt"
    people = "./data/real/image/"
    cloth = "./data/real/cloth/"
    get_pair(people,cloth,note)
    
    main_real_tom()

futurama()





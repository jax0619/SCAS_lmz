from __future__ import print_function, division
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
import torch
import torch.nn as nn
import os
import numpy as np
import time
from utils import adjust_learning_rate, setup_seed
from model import SCSANet
from dataset import CrowdCountingDataset
# %matplotlib inline

################################################################################
# configuration
################################################################################
# set random seed for reproducibility
manualSeed = 1
# manualSeed = random.randint(1, 10000) # use if you want new results
# print("Random Seed: ", manualSeed)
setup_seed(manualSeed)
# choose to run on cpu or cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# create a directory to store the model
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
    os.mkdir('checkpoint/models')


################################################################################
# train SCASNet model to generate density map
################################################################################
def train():
    """
        train model
    """
    # set hyperparameter
    TRAIN_IMG_DIR = 'datasets/part_A_final/train_data/images' # the directory path for storing training set images
    TEST_IMG_DIR = 'datasets/part_A_final/test_data/images' # the directory path for storing test set images
    LR = 1e-4 # learning rate
    EPOCH = 1000 # training epoch
    BATCH_SIZE = 30 # batch size
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    resume = False  # whether to breakpoint training
    workers = 4  # number of workers for dataloader
    hyper_param_D = 0.001 # the weight parameter of the loss function
    lc_loss1 = 1000
    cross = 1

    # best MAE, MSE
    # BEST_MAE = float("inf")
    BEST_MAE = 500
    # BEST_MSE = float("inf")
    BEST_MSE = 500

    # load data
    MEAN = [0.485, 0.456, 0.406] # mean
    STD = [0.229, 0.224, 0.225] # std
    normalize = transforms.Normalize(
        mean=MEAN,
        std=STD
    )
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize]
    )
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize]
    )
    # define trainloader
    train_dataset = CrowdCountingDataset(TRAIN_IMG_DIR, transforms = train_transform, scale = 8, mode = 'train')
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers=workers)
    # define valloader
    val_dataset = CrowdCountingDataset(TEST_IMG_DIR, transforms = val_transform, scale = 8, mode = 'test')
    val_loader = DataLoader(val_dataset, batch_size = 1, num_workers=workers)

    # define model
    model = SCASNet().float()
    model = model.to(device)

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # breakpoint training, load model weights
    if resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no Checkpoint directory found!'
        state = torch.load('checkpoint/models/ckpt.pth')
        model.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optim'])
        start_epoch = state['epoch']
        BEST_MAE = state['mae']
        BEST_MSE = state['mse']

    # loss function
    mseloss = nn.MSELoss(reduction='sum').to(device)
    cosloss = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
    def cal_lc_loss(output, target, sizes=(1, 2, 4)):
        criterion_L1 = nn.L1Loss()
        Lc_loss = None
        for s in sizes:
            pool = nn.AdaptiveAvgPool2d(s)
            est = pool(output)
            gt = pool(target)
            if Lc_loss:
                Lc_loss += criterion_L1(est, gt)
            else:
                Lc_loss = criterion_L1(est, gt)
        return Lc_loss

    # train model
    for epoch in range(start_epoch, EPOCH):
        print("####################################################################################")
        # learning rate scheduling strategy
        adjust_learning_rate(optimizer, epoch)
        print('Learning rate is {}'.format(optimizer.param_groups[0]['lr']))
        ############################
        # train
        ############################
        start_time = time.time()
        # train mode
        model.train()
        # loss
        sum_loss = 0.0
        sum_att_loss = 0.0
        sum_den_loss = 0.0
        sum_lc_loss = 0.0
        sum_losscross = 0.0
        # number of iterations
        cnt = 0
        for data in train_loader:
            cnt += 1

            # load data
            image, gt_density = data
            image, gt_density = image.float(), gt_density.float()
            image, gt_density = image.to(device), gt_density.to(device)

            # gradient zero
            optimizer.zero_grad()

            # forward and backward propagation
            pr_density, attention_map_1, attention_map_2, attention_map_3 = model(image)
            attention_loss = 0.
            for attention_map in (attention_map_1, attention_map_2, attention_map_3):
                # print("attention_map:"+str(attention_map))
                attention_map_sum = attention_map[:, 0:1] + attention_map[:, 1:2] + attention_map[:, 2:3] +\
                                    attention_map[:, 3:4]
                # print("attention_map_sum:"+str(attention_map_sum))
                attention_loss_temp = 0.
                for i in range(4):
                    attention_loss_temp += torch.sum(cosloss(attention_map[:, i:(i+1)].contiguous().view(image.size(0), -1),
                                                  ((attention_map_sum-attention_map[:, i:(i+1)])/3).contiguous().view(image.size(0), -1))) / image.size(0)
                attention_loss += (attention_loss_temp / 4)
            attention_loss /= 3
            density_loss = mseloss(pr_density, gt_density) / image.size(0)
            lc_loss = cal_lc_loss(pr_density, gt_density)
            import torch.nn.functional as F

            loss = density_loss + hyper_param_D*attention_loss + lc_loss1*lc_loss + cross*losscross
            # print(losscross)
            loss.backward()

            # gradient update
            optimizer.step()
            sum_loss += loss.item()
            sum_att_loss += attention_loss.item()
            sum_den_loss += density_loss.item()
            sum_lc_loss += lc_loss.item()
            sum_losscross += losscross.item()

            # print log
            if cnt % 5 == 0 or cnt == len(train_loader):
                print('[%d/%d]--[%d/%d]\tLoss: %.4f\tAtt_Loss: %.4f\tDen_Loss: %.4f\tlc_loss: %.4f\tlosscross: %.4f'
                        % (epoch + 1, EPOCH, cnt, len(train_loader), sum_loss / cnt,
                           sum_att_loss / cnt, sum_den_loss / cnt, sum_lc_loss / cnt, losscross / cnt))
        t_loss = sum_loss / cnt
        # save model
        state = {
            'net': model.state_dict(),
            'optim': optimizer.state_dict(),
            'epoch': epoch,
            'mae': BEST_MAE,
            'mse': BEST_MSE
        }
        torch.save(state, 'checkpoint/models/ckpt.pth')
        print('train time:%.2f[second]'%(time.time() - start_time))
        ############################
        # test
        ############################
        # test mode
        start_time1 = time.time()
        model.eval()
        # loss
        mae = 0.0
        mse = 0.0
        # number of iterations
        cnt = 0
        with torch.no_grad():
            for data in val_loader:
                cnt += 1

                # load data
                image, gt, gt_density = data
                image, gt_density = image.float(), gt_density.float()
                image, gt_density = image.to(device), gt_density.to(device)

                # forward and backward propagation
                pr_density, attention_map_1, attention_map_2, attention_map_3 = model(image)

                # record real results and predicted results
                pr_density = pr_density.cpu().detach().numpy()
                gt_density = gt_density.cpu().detach().numpy()
                pr = np.sum(pr_density)
                gt = np.sum(gt_density)
                mae += np.abs(gt - pr)
                mse += np.abs(gt - pr) ** 2

        # calculate loss
        mae_loss = mae / cnt
        mse_loss = np.sqrt(mse / cnt)
        # update best mse, mae
        BEST_MSE = min(BEST_MSE, mse_loss)
        if BEST_MAE > mae_loss:
          BEST_MAE = mae_loss
          # save best model
          state = {
              'net': model.state_dict(),
              'optim': optimizer.state_dict(),
              'epoch': epoch,
              'mae': BEST_MAE,
              'mse': BEST_MSE
          }
          torch.save(state, 'checkpoint/models/ckpt_best.pth')

        # print log
        print('[%d/%d]\tval time: %.2f[second]\tLoss_T: %.4f\tMAE: %.4f\tMSE: %.4f\tBEST_MAE: %.4f\tBEST_MSE: %.4f'
              % (epoch + 1, EPOCH, time.time() - start_time1, t_loss, mae_loss, mse_loss, BEST_MAE, BEST_MSE))


################################################################################
# main function
################################################################################
if __name__ == '__main__':
    # train model
    train()

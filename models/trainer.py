import math

from torch.autograd import Variable
import torch.nn.functional as F
import torch
from datetime import datetime
import cv2


def get_one_hot(num, gt):

    a = torch.ones([1, 2, num]).cuda()

    a[:, 0, gt.view(num) == 1] = 1
    a[:, 1, gt.view(num) == 1] = 0

    a[:, 0, gt.view(num) == 0] = 0
    a[:, 1, gt.view(num) == 0] = 1


    return a



def similarity(input, target):

    target_1_4 = F.interpolate(target, scale_factor=0.25, mode="nearest").view(1, 1, -1)
    target_1_4[target_1_4 > 0.5] = 1
    target_1_4[target_1_4 <= 0.5] = 0
    target_1_4 = get_one_hot(target_1_4.shape[2], target_1_4)
    target_1_4 = torch.matmul(target_1_4.permute(0, 2, 1), target_1_4)


    loss_1_4 = F.binary_cross_entropy_with_logits(input, target_1_4)


    return loss_1_4



class Trainer(object):

    def __init__(self, cuda, model_rgb,
                 model_focal, model_fuse, optimizer_rgb,
                 optimizer_focal,optimizer_fuse,
                 train_loader, max_iter, snapshot, outpath, sshow, size_average=False):
        self.cuda = cuda
        self.model_rgb = model_rgb
        self.model_focal = model_focal
        self.model_fuse = model_fuse
        self.optim_rgb = optimizer_rgb
        self.optim_focal = optimizer_focal
        self.optim_fuse = optimizer_fuse
        self.train_loader = train_loader
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.snapshot = snapshot
        self.outpath = outpath
        self.sshow = sshow
        self.size_average = size_average



    def train_epoch(self):


        for batch_idx, (image, target, focal) in enumerate(self.train_loader):


            iteration = batch_idx + self.epoch * len(self.train_loader)


            if iteration == 150000 or iteration == 180000:
                self.adjust_lr(self.optim_rgb, decay_rate=0.1)
                self.adjust_lr(self.optim_focal, decay_rate=0.1)
                self.adjust_lr(self.optim_fuse,  decay_rate=0.1)



            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue # for resuming
            self.iteration = iteration
            if self.iteration >= self.max_iter:
                break
            if self.cuda:
                image, target, focal = image.cuda(), target.cuda(), focal.cuda()

            basize, dime, height, width = focal.size() # 2*36*256*256

            focal_num = focal.shape[1] // 3
            focal = torch.chunk(focal, focal_num, dim=1)
            focal = torch.cat(focal, dim=0)


            self.optim_rgb.zero_grad()
            self.optim_focal.zero_grad()
            self.optim_fuse.zero_grad()

            lowlevel_feature_rgb, rgb_feature = self.model_rgb(image)
            focal_feature = self.model_focal(focal)
            pred0, pred1, pred2, pred3, pred4 = self.model_fuse(rgb_feature, focal_feature, lowlevel_feature_rgb)

            loss0 = F.binary_cross_entropy_with_logits(pred0, target)
            loss1 = F.binary_cross_entropy_with_logits(pred1, target)
            loss2 = F.binary_cross_entropy_with_logits(pred2, target)
            loss3 = F.binary_cross_entropy_with_logits(pred3, target)
            loss4 = F.binary_cross_entropy_with_logits(pred4, target)

            loss = (loss0 + loss1 + loss2 + loss3 + loss4) / 5

            loss.backward()
            self.optim_rgb.step()
            self.optim_focal.step()
            self.optim_fuse.step()



            if iteration % 10 == 0:


                print('{} Epoch [{:03d}], Step [{:04d}/{:04d}], total loss: {:0.4f}'.format(datetime.now(), self.epoch, iteration, self.max_iter, loss.data))

            if iteration % 50 == 0:
                pred1 = F.sigmoid(pred1)
                pred1 = pred1[0][0].cpu().data.numpy()
                cv2.imwrite("show/{}_1.png".format(iteration), 255 * pred1)




            if (iteration+1) == self.max_iter:
                savename = ('%s/snapshot_iter_%d.pth' % (self.outpath, iteration+1))
                torch.save(self.model_rgb.state_dict(), savename)
                print('save: (snapshot: %d)' % (iteration+1))

                savename_focal = ('%s/focal_snapshot_iter_%d.pth' % (self.outpath, iteration+1))
                torch.save(self.model_focal.state_dict(), savename_focal)
                print('save: (snapshot_focal: %d)' % (iteration+1))

                savename_fuse = ('%s/fuse_snapshot_iter_%d.pth' % (self.outpath, iteration+1))
                torch.save(self.model_fuse.state_dict(), savename_fuse)
                print('save: (snapshot_fuse: %d)' % (iteration+1))



    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))

        for epoch in range(max_epoch):
            print(epoch)
            print(self.optim_rgb.param_groups[0]["lr"])
            self.epoch = epoch
            self.train_epoch()
            if epoch != 0 and epoch % 1 == 0:

                savename = ('%s/snapshot_epoch_%d.pth' % (self.outpath, epoch))
                torch.save(self.model_rgb.state_dict(), savename)


                savename_focal = ('%s/focal_snapshot_epoch_%d.pth' % (self.outpath, epoch))
                torch.save(self.model_focal.state_dict(), savename_focal)


                savename_fuse = ('%s/fuse_snapshot_epoch_%d.pth' % (self.outpath, epoch))
                torch.save(self.model_fuse.state_dict(), savename_fuse)

            if self.iteration >= self.max_iter:
                break

    def adjust_lr(self, optimizer, decay_rate=0.1):
        decay = decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay
        print("lr=", optimizer.param_groups[0]["lr"])

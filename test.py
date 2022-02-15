
import os
import argparse
from torch.utils.data import DataLoader
import torch
from dataset.dataset_loader import MyTestData
from models.backbone import RGBNet, FocalNet
from models.fusenetwork import FuseNet
import torchvision
from models.trainer import Trainer
import cv2
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser=argparse.ArgumentParser()
parser.add_argument('--snapshot_root', type=str, default='./snapshot/', help='path to snapshot')
parser.add_argument('--test_dataroot', type=str, default="the dir of testset", help='path to test data')
parser.add_argument('--salmap_root', type=str, default='./sal_map', help='path to saliency map')
args = parser.parse_args()


save_list = ["_epoch_10.pth", "_epoch_11.pth", "_epoch_12.pth", "_epoch_13.pth", "_epoch_14.pth", "_epoch_15.pth",
             "_epoch_16.pth", "_epoch_17.pth", "_epoch_18.pth", "_iter_200000.pth"]

for save_name in save_list:

    model_rgb = RGBNet()
    model_focal = FocalNet()
    model_fuse = FuseNet()

    model_rgb.load_state_dict(torch.load(os.path.join(args.snapshot_root, 'snapshot' + save_name)))
    model_focal.load_state_dict(torch.load(os.path.join(args.snapshot_root, 'focal_snapshot' + save_name)))
    model_fuse.load_state_dict(torch.load(os.path.join(args.snapshot_root, 'fuse_snapshot'+ save_name)))



    model_rgb = model_rgb.cuda()
    model_focal = model_focal.cuda()
    model_fuse = model_fuse.cuda()

    test_loader = torch.utils.data.DataLoader(MyTestData(root=args.test_dataroot, transform=True), batch_size=1, shuffle=False)
    with torch.no_grad():
        for id, (img_file, image, focal) in enumerate(test_loader):
            image, focal = image.cuda(), focal.cuda()

            basize, dime, height, width = focal.size()  # 2*36*256*256

            focal_num = focal.shape[1] // 3
            focal = torch.chunk(focal, focal_num, dim=1)
            focal = torch.cat(focal, dim=0)

            lowlevel_feature_rgb, rgb_feature = model_rgb(image)
            focal_feature = model_focal(focal)
            _, _, _, _, pred = model_fuse(rgb_feature, focal_feature, lowlevel_feature_rgb)

            pred = F.sigmoid(pred)
            pred = pred[0][0].cpu().data.numpy()
            pred = 255 * pred

            dataset = img_file[0].split("/")[-4]
            image_name =  img_file[0].split("/")[-1]

            print(os.path.join(args.salmap_root, dataset, image_name))
            if not os.path.exists(os.path.join(args.salmap_root, dataset, os.path.abspath(os.path.dirname(__file__)).split("/")[-1]+save_name)):
                os.makedirs(os.path.join(args.salmap_root, dataset, os.path.abspath(os.path.dirname(__file__)).split("/")[-1]+save_name))



            cv2.imwrite(os.path.join(args.salmap_root, dataset, os.path.abspath(os.path.dirname(__file__)).split("/")[-1]+save_name, image_name[:-3] + "png"), pred)


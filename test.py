import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time



#from lib.dataset.train_dataset_fof import get_dataloader
from lib.network.HRNet import HRNetV2_W32 as network
from lib.utils.utils import toDevice
from lib.utils.utils import Recon


def test(args):
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:%d"%args.gpu_id)
    print("using device:", device)

    e = Recon(device)

    net = network().to(device)
    if args.ckpt == "latest":
        ckpt_list = sorted(os.listdir(os.path.join("ckpt", args.name)))
        if len(ckpt_list) > 0:
            ckpt_path = os.path.join("ckpt", args.name, ckpt_list[-1])
            print('Resuming from ', ckpt_path)
            state_dict = torch.load(ckpt_path)
            net.load_state_dict(state_dict["net"])
            del state_dict
        else:
            print("No checkpont !")
            exit()
    else:
        print('Resuming from ', args.ckpt)
        state_dict = torch.load(args.ckpt)
        net.load_state_dict(state_dict["net"])
        del state_dict

    # test
    input_dir = sorted(os.listdir(args.input))

    net.eval()
    with torch.no_grad():
        for d in input_dir:
            img = cv2.imread(os.path.join(args.input, d), -1)
            img = cv2.resize(img, (512,512))
            mask = img[:,:,3:4]
            img = img[:,:,:3]
            img = torch.from_numpy(img.transpose((2,0,1)))[None]
            mask = torch.from_numpy(mask.transpose((2,0,1)))[None]
            img = img.to(device)
            mask = mask.to(device) > 127


            predicted = net((img/127.5 - 1)*mask)*mask
            predicted = torch.clip(predicted, -1, 1)
            print(predicted.shape, predicted.max(), predicted.min())
            image_B = predicted.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

            cv2.imwrite(os.path.join("output", d), np.uint8((image_B + 1) * 127))

            print(np.uint8((image_B + 1) * 127).max(), np.uint8((image_B + 1) * 127).min())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",   type=str, default="base",   help="name of the experiment")
    parser.add_argument("--gpu_id", type=int, default=7,        help="gpu id for cuda")
    parser.add_argument("--ckpt",   type=str, default="latest", help="path of the checkpoint")
    parser.add_argument("--input",  type=str, default="input",  help="path of the input_dir")
    parser.add_argument("--output", type=str, default="output", help="path of the output_dir")

    parser.add_argument('--T', type=float, default=1000)
    parser.add_argument('--beta_1', type=float, default=1e-4)
    parser.add_argument('--beta_T', type=float, default=0.02)

    args = parser.parse_args()
    test(args)
    

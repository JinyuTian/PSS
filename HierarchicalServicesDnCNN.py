from utils import save_checkpoint, AverageMeter, accuracy
import argparse
import os
from torch.nn.modules.loss import _Loss
from Models import DnCNN, OriDnCNN
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import vgg
import numpy as np
import warnings

warnings.filterwarnings('ignore')

model_names = sorted(name for name in vgg.__dict__
                     if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--ResumePath', default='', type=str, metavar='Path of pretrained DnCNN')
parser.add_argument('--layers', type=list,help = 'layers to be encrypted')
parser.add_argument('-sigma', type=float,help='nosie level of DnCNN')
parser.add_argument('-Ratio', type=float,help = 'percentages of encrypted parameters')
parser.add_argument('-Repeat', type=int,help='repeat times')
parser.add_argument('-M', type=float,help='number of permissions')
parser.add_argument('-Key', type=int,help='key to generate random number')
parser.add_argument('-ImagePath', type=str,help='test image path')
parser.add_argument('-importance_resume',type=str, help = 'Path to save importance of parameters')


parser.set_defaults(sigma=50.0)
parser.set_defaults(Ratio=0.02)
parser.set_defaults(M=5)
parser.set_defaults(Key=5)
parser.set_defaults(Repeat=1)
parser.set_defaults(ResumePath='PretrainedModel/DnCNN.pth.tar')
parser.set_defaults(ImagePath='./data/DnCNNTestOne/test2.png')
parser.set_defaults(layers=[5, 8, 11])
parser.set_defaults(importance_resume='Importance')

best_prec1 = 100
writer = None
time_acc = [(0, 0, 0)]
total_steps = 0
exp_flops, exp_l0 = [], []


def main():
    global args, best_prec1, writer, time_acc, total_steps, exp_flops, exp_l0
    args = parser.parse_args()
    with torch.no_grad():
        for m in range(args.M+1):
            #### Load DnCNN ####
            model = DnCNN(image_channels=3)
            checkpoint = torch.load(args.ResumePath)
            model.load_state_dict(checkpoint['state_dict'])
            Es = Select(model,args)
            model,STD,MEAN = Encrypt(model, Es, args.Key,args.M-1)
            model = Decrypt(model, Es, args.Key, m, STD, MEAN)
            PSNR = ValOne(model, args.ImagePath)
            print('The PSNR of decrypted model (Permission Level = {} % parameters are encrypted) is {}'.format(m, PSNR))

def Decrypt(model, Es, SEED,m,STD,MEAN):
    model.cpu()
    if not m == 0:
        for layerID in Es.keys():
            E_locations = Es[layerID]
            model.layers[layerID].De_DPRM(E_locations,SEED,STD[layerID],MEAN[layerID],m-1)
    return model

def Encrypt(model, Es, SEED,m):
    model.cpu()
    STD = {}
    MEAN = {}
    for layerID in Es.keys():
        E_locations = Es[layerID][m]
        EN,A,std, mean,ori = model.layers[layerID].DPRM(E_locations, 'Gaussian', seed=SEED)
        STD[layerID] = std
        MEAN[layerID] = mean
    return model,STD,MEAN

def Select(model,args):
    #### ExperimentLog
    EncryptionLocaltion = {}
    for i, layerID in enumerate(args.layers):
        path = os.path.join(args.importance_resume, 'DnCNN_layer_{}.pth.tar'.format(layerID))
        checkpoint = torch.load(path)
        importance = checkpoint['importance']
        model.layers[layerID].qz_loga.copy_(importance)
        PSS_E = model.layers[layerID].EncryptLocation(args.Ratio, 'PSS', args.M)
        EncryptionLocaltion[layerID] = PSS_E

    return EncryptionLocaltion

def ValOne(model, ImagePath):
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'qz_loga'):
            layer.reset_to_10()
    if not os.path.exists('./OutputImages'):
        os.mkdir('./OutputImages')
    save_path = os.path.join('./OutputImages',ImagePath.split('/')[-1].split('.png')[0])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model.eval()
    model.cuda()
    import cv2
    x = cv2.imread(ImagePath)
    Tx = x.astype('float32') / 255.0
    Tx = Tx.transpose((2, 0, 1))
    batch_x = torch.from_numpy(Tx)
    torch.manual_seed(3)
    noise = torch.randn(batch_x.shape).mul_(args.sigma / 255.0)
    batch_y = batch_x + noise
    batch_x = batch_x.unsqueeze(0)
    batch_y = batch_y.unsqueeze(0)
    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
    denoise = model(batch_y)
    DenoiseImage = denoise.detach().cpu().numpy().squeeze(0) * 255.0
    OriImage = batch_x.detach().cpu().numpy().squeeze(0) * 255.0
    NoiseImage = batch_y.detach().cpu().numpy().squeeze(0) * 255.0
    T_psnr_Denoise = calculate_psnr(OriImage, DenoiseImage)
    T_psnr_Noise = calculate_psnr(OriImage, NoiseImage)
    DenoiseImage = DenoiseImage.transpose((1,2,0))
    NoiseImage = NoiseImage.transpose((1,2,0))
    cv2.imwrite(os.path.join(save_path,'M{}_{:.2f}.png'.format(args.M,T_psnr_Denoise)),DenoiseImage)
    cv2.imwrite(os.path.join(save_path,'noise_{:.2f}.png'.format(T_psnr_Noise)),NoiseImage)
    return T_psnr_Denoise

def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))




if __name__ == '__main__':
    main()

import os
from dataset import *
import torch
import torch.nn as nn
import random
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import time
from torch.autograd import Variable
from model import *
#from test_model import *
import numpy as np
import argparse

def range_compressor_tensor(x):
    const_1 = torch.from_numpy(np.array(1.0)).cuda()
    const_5000 = torch.from_numpy(np.array(5000.0)).cuda()
    return (torch.log(const_1 + const_5000 * x)) / torch.log(const_1 + const_5000)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])#,(SSIM/Img.shape[0])

def batch_PSNR_SSIM(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    SSIM = 0
#    print(Img.shape,Iclean.shape)
    for i in range(Img.shape[0]):
#        print("Iclean[i, :, :, :] = ",Iclean[i, :, :, :].transpose(1,2,0).shape)
        PSNR += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
        SSIM += ssim(Iclean[i, :, :, :].transpose(1,2,0),
                     Img[i, :, :, :].transpose(1,2,0),
                     data_range=data_range, multichannel=True)
    return (PSNR/Img.shape[0]),(SSIM/Img.shape[0])

def loss_sobel_v3(y_pred, y_true):
    mae = loss_mae
    mae_loss = mae(y_pred, y_true)
    sobel_pred = sobel_filter_v3(y_pred)*0.25
    sobel_true = sobel_filter_v3(y_true)*0.25
    dx_loss = mae(sobel_pred[:, :, :, :, 0], sobel_true[:, :, :, :, 0])
    dy_loss = mae(sobel_pred[:, :, :, :, 1], sobel_true[:, :, :, :, 1])
    dr_loss = mae(sobel_pred[:, :, :, :, 2], sobel_true[:, :, :, :, 2])
    dl_loss = mae(sobel_pred[:, :, :, :, 3], sobel_true[:, :, :, :, 3])
    return mae_loss + dx_loss + dy_loss + dr_loss + dl_loss
def sobel_filter_v3(x):
    x_in = x
    x_in = F.pad(x_in, (1, 1, 1, 1), 'reflect')

    sobel1 = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    sobel2 = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel3 = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
    sobel4 = [[0, -1, -2], [1, 0, -1], [2, 1, 0]]

    sobel1 = torch.FloatTensor(sobel1).unsqueeze(0).unsqueeze(0)
    sobel2 = torch.FloatTensor(sobel2).unsqueeze(0).unsqueeze(0)
    sobel3 = torch.FloatTensor(sobel3).unsqueeze(0).unsqueeze(0)
    sobel4 = torch.FloatTensor(sobel4).unsqueeze(0).unsqueeze(0)

    sobel1 = torch.repeat_interleave(sobel1, repeats=3, dim=0)
    sobel2 = torch.repeat_interleave(sobel2, repeats=3, dim=0)
    sobel3 = torch.repeat_interleave(sobel3, repeats=3, dim=0)
    sobel4 = torch.repeat_interleave(sobel4, repeats=3, dim=0)

    sobel1 = sobel1.cuda()
    sobel2 = sobel2.cuda()
    sobel3 = sobel3.cuda()
    sobel4 = sobel4.cuda()

    y_1 = F.conv2d(x_in, sobel1, groups=3)
    y_2 = F.conv2d(x_in, sobel2, groups=3)
    y_3 = F.conv2d(x_in, sobel3, groups=3)
    y_4 = F.conv2d(x_in, sobel4, groups=3)

    shape = torch.zeros_like(x).unsqueeze(-1)
    y = torch.repeat_interleave(shape, repeats=4, dim=-1)

    y[:, :, :, :, 0] = y_1
    y[:, :, :, :, 1] = y_2
    y[:, :, :, :, 2] = y_3
    y[:, :, :, :, 3] = y_4

    return y
def loss_mae(y_pred, y_true):
    return nn.L1Loss()(y_pred, y_true)


def train(train_loader, model, lr=0.01, epoch=1,img_size=None):
    model.train()
    start = time.time()
#    loss = nn.L1Loss()
    loss = loss_sobel_v3
    img_size = img_size
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for step, sample in enumerate(train_loader):
        batch_x1, batch_x2, batch_x3, batch_x4 = sample['input1'], sample['input2'], sample['input3'], sample[
            'label']
        batch_x1, batch_x2, batch_x3, batch_x4 = Variable(batch_x1).cuda(), Variable(batch_x2).cuda(), Variable(
            batch_x3).cuda(), Variable(batch_x4).cuda()

        batch_x1 = torch.as_tensor(batch_x1, dtype=torch.float32)
        batch_x2 = torch.as_tensor(batch_x2, dtype=torch.float32)
        batch_x3 = torch.as_tensor(batch_x3, dtype=torch.float32)
        for i in range(256 // img_size):
            _batch_x1 = batch_x1[:, :, i * img_size:(i + 1) * img_size, i * img_size:(i + 1) * img_size]
            _batch_x2 = batch_x2[:, :, i * img_size:(i + 1) * img_size, i * img_size:(i + 1) * img_size]
            _batch_x3 = batch_x3[:, :, i * img_size:(i + 1) * img_size, i * img_size:(i + 1) * img_size]
            _batch_x4 = batch_x4[:, :, i * img_size:(i + 1) * img_size, i * img_size:(i + 1) * img_size]
            pre = model(_batch_x1, _batch_x2, _batch_x3)
            pre = range_compressor_tensor(pre)
            pre = torch.clamp(pre, 0., 1.)
            g_loss = loss(pre, _batch_x4)
            optimizer.zero_grad()
            g_loss.backward()
            optimizer.step()
            PSNR = batch_PSNR(torch.clamp(pre, 0., 1.), _batch_x4, 1.)
#            wandb.log({"loss": g_loss.item(), "lr": lr, "batch_psnr": PSNR})
            print('epoch:{:d} {:d}/{:d}-->lr:{:f} loss:{:6f} PSNR:{:f} time:{:.4f}s'.format(
                epoch, step + 1, len(train_loader), lr, g_loss.item(), PSNR, time.time() - start), end='\r')

def makeimg1(img,add_h,add_w):
    img = img.squeeze().permute(1,2,0)
    img = img.numpy()
    img = cv2.copyMakeBorder(img, add_h, 0, add_w, 0, borderType=cv2.BORDER_REFLECT_101)
    img = torch.as_tensor(img,dtype=torch.float32)
    img = img.permute(2,0,1).unsqueeze(dim=0)
    return img


def makeimg(img,add_h,add_w):
    img = img.squeeze().permute(1,2,0)
    a = img[:,:,:3]
    b = img[:,:,3:]
#    print(a.shape(),b.shape())
    a = a.numpy()
    b = b.numpy()
    a = cv2.copyMakeBorder(a, add_h, 0, add_w, 0, borderType=cv2.BORDER_REFLECT_101)
    b = cv2.copyMakeBorder(b, add_h, 0, add_w, 0, borderType=cv2.BORDER_REFLECT_101)
    img = np.concatenate((a,b),axis=2)
    img = torch.as_tensor(img,dtype=torch.float32)
    img = img.permute(2,0,1).unsqueeze(dim=0)
#    print(img.size())
    return img

def test(test_data, test_loader, model, best_psnr=0, fail=0, epoch=0,
         img_size = None,name=None,block = None,data_name = None):
    model.eval()
    save_name = str(data_name)+'_'+name+'_'+str(block)+'_'+str(img_size)+'.pth'
    val_psnr = 0
    start = time.time()
    with torch.no_grad():
        for step, sample in enumerate(test_loader):
            batch_x1, batch_x2, batch_x3, batch_x4 = sample['input1'], sample['input2'], sample['input3'], sample[
                'label']
            h, w = batch_x1.shape[2], batch_x1.shape[3]
            add_h, add_w = 16 - h % 16, 16 - w % 16
            batch_x1 = makeimg(batch_x1,add_h,add_w)
            batch_x2 = makeimg(batch_x2,add_h,add_w)
            batch_x3 = makeimg(batch_x3,add_h,add_w)

            batch_x1 = torch.as_tensor(batch_x1, dtype=torch.float32)
            batch_x2 = torch.as_tensor(batch_x2, dtype=torch.float32)
            batch_x3 = torch.as_tensor(batch_x3, dtype=torch.float32)
            batch_x1, batch_x2, batch_x3, batch_x4 = Variable(batch_x1).cuda(), Variable(batch_x2).cuda(), Variable(
                batch_x3).cuda(), Variable(batch_x4).cuda()

            pre = model(batch_x1, batch_x2, batch_x3)
            pre = pre[:, :,add_h:, add_w:]
            batch_x4 = batch_x4.cpu()
            batch_x4 = (np.log(1 + 5000 * batch_x4)) / np.log(1 + 5000)
            pre = range_compressor_tensor(pre)
            PSNR = batch_PSNR(torch.clamp(pre, 0., 1.), batch_x4, 1.)
            val_psnr = val_psnr + PSNR * batch_x4.shape[0]
    end = time.time()
    val_psnr /= test_data.__len__()
    if best_psnr > val_psnr:
        fail += 1
    else:
        fail = 0
        best_psnr = val_psnr
        torch.save(model.state_dict(), './p19_model/' + save_name)
    print('epoch:{:d} psnr:{:.6f} best_psnr:{:.6f} fail:{:d} time:{:.4f}s save:{}'
          .format(epoch, val_psnr, best_psnr,fail, end - start,save_name))
    if fail >= 4:
        return 1, 0, best_psnr
    else:
        return 0, fail, best_psnr


def test_psnr_ssim(test_data, test_loader, model):
    model.eval()
    val_psnr_u,val_psnr_l = 0,0
    val_ssim_u,val_ssim_l = 0,0
    with torch.no_grad():
        for step, sample in enumerate(test_loader):
            batch_x1, batch_x2, batch_x3, batch_x4 = sample['input1'], sample['input2'], sample['input3'], sample[
                'label']
            h,w = batch_x1.shape[2],batch_x1.shape[3]
            h = h-h%16
            w = w-w%16
            batch_x1 = batch_x1[:,:,:h,:w]
            batch_x2 = batch_x2[:, :, :h, :w]
            batch_x3 = batch_x3[:, :, :h, :w]
            batch_x4 = batch_x4[:, :, :h, :w]

            batch_x1 = torch.as_tensor(batch_x1, dtype=torch.float32)
            batch_x2 = torch.as_tensor(batch_x2, dtype=torch.float32)
            batch_x3 = torch.as_tensor(batch_x3, dtype=torch.float32)
            batch_x1, batch_x2, batch_x3, batch_x4 = Variable(batch_x1).cuda(), Variable(batch_x2).cuda(), Variable(
                batch_x3).cuda(), Variable(batch_x4).cuda()

            pre = model(batch_x1, batch_x2, batch_x3)
            PSNR_l, SSIM_l = batch_PSNR_SSIM(torch.clamp(pre, 0., 1.), batch_x4, 1.)
            pre = range_compressor_tensor(pre)
            batch_x4 = batch_x4.cpu()
            batch_x4 = (np.log(1 + 5000 * batch_x4)) / np.log(1 + 5000)

            PSNR_u,SSIM_u = batch_PSNR_SSIM(torch.clamp(pre, 0., 1.), batch_x4, 1.)
            print("PSNR == ", PSNR_u, end='\r')
            val_psnr_l = val_psnr_l + PSNR_l * batch_x4.shape[0]
            val_psnr_u = val_psnr_u + PSNR_u * batch_x4.shape[0]
            val_ssim_l = val_ssim_l + SSIM_l * batch_x4.shape[0]
            val_ssim_u = val_ssim_u + SSIM_u * batch_x4.shape[0]
#    end = time.time()
    val_psnr_l /= test_data.__len__()
    val_psnr_u /= test_data.__len__()
    val_ssim_u /= test_data.__len__()
    val_ssim_l /= test_data.__len__()
    print("psnr_l,psnr_u,ssim_u,ssim_l = ",val_psnr_l,val_psnr_u,val_ssim_u,val_ssim_l)

def main():
    parser = argparse.ArgumentParser(description="setting")
    parser.add_argument("--path",type=str,default="../h5_data_raw_ldr=uint16_hdr=float32/")
    parser.add_argument("--num_cuda",type=str,default='0')
    parser.add_argument("--img_size",type=int,default=128)
    parser.add_argument("--nblock",type=int,default=3)
    parser.add_argument("--batch_size",type=int,default=12)
    parser.add_argument("--dataset_name",type=str,default='kala')
    parser.add_argument("--nfeat",type=int,default=64)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.num_cuda
    model = HFT(6,args.nfeat,args.nblock).cuda()
    name = model._get_name()
    lr = [1e-4, 5e-5, 2.5e-5, 1e-5, 5e-6, 1e-6]
    lr_index = 0
    batch_size =args.batch_size
    num_workers = 16
    img_size = args.img_size
    path1 = args.path
    train_data = Dataset_h5(path1 + "Training/")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_data = Dataset_h5_test(path1 + "Test/")
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=num_workers, drop_last=True)
    best_psnr = -999
    fail = 0
    for epoch in range(160):
        if epoch != 0:
            train(train_loader, model, lr[lr_index], epoch,img_size)
        add, fail, best_psnr = test(test_data, test_loader, model, best_psnr,fail,epoch,
                                    img_size,name,args.nblock,args.dataset_name)
        lr_index += add
        if lr_index == len(lr):
            model.load_state_dict(torch.load("./p19_model/p19_test1_p_model5_3_128.pth"))
            test_psnr_ssim(test_data, test_loader, model)
            print("model finish!")
            break

if __name__ == "__main__":
    main()

import os
from model import CSRNet
from utils import save_checkpoint
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import argparse
import json
import dataset
import time

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('--train_json',
                    help='path to train json',
                    default='./train_list.json')
parser.add_argument('--test_json',
                    help='path to test json',
                    default='./test_list.json')

parser.add_argument('--pre', '-p', default='', type=str,
                    help='path to the pretrained model',)

parser.add_argument('--gpu', type=str,
                    help='GPU id to use.',
                    default='cuda:0')

parser.add_argument('--task', type=str,
                    help='task id to use.', default='01_')

def main():
    
    global args, best_prec1
    
    best_prec1 = 1e6
    
    args = parser.parse_args()
    args.original_lr = 1e-5
    args.lr = 1e-5
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 400
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30

    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)
    

    model = CSRNet()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # criterion = nn.MSELoss(reduction='sum').to(device)
    # criterion = nn.L1Loss(reduction='mean').to(device)
    criterion = nn.SmoothL1Loss(reduction='mean', beta=0.05)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
            
    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch)
        
        train(train_list, model, criterion, optimizer, epoch, device)
        prec1 = validate(val_list, model, criterion, device)
        
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.task)

def train(train_list, model, criterion, optimizer, epoch, device):
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]
                                                ),
                   ]), 
                       train=True, 
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    
    model.train()
    end = time.time()


    for i,(img, target)in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.to(device)
        # img = Variable(img)
        output = model(img)
        

        target = target.type(torch.FloatTensor).unsqueeze(0).to(device)
        # target = Variable(target)

        if i == 0:
            # Сохраняем три изображения (img, target, output) в один файл
            plt.figure(figsize=(12, 4))

            # Исходное изображение
            plt.subplot(1, 3, 1)
            img_cpu = img.cpu().squeeze(0).permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_display = std * img_cpu + mean
            img_display = np.clip(img_display, 0, 1)
            plt.imshow(img_display)
            plt.title('Image')
            plt.axis('off')

            max_abs = 1e-5
            # Целевая карта плотности
            plt.subplot(1, 3, 2)
            target_cpu = target.cpu().squeeze(0).squeeze(0).numpy()
            plt.imshow(target_cpu, cmap='jet',)
            plt.title(f'Target (sum: {target_cpu.sum():.0f})')
            plt.axis('off')
            plt.colorbar()

            # Выход сети
            plt.subplot(1, 3, 3)
            output_cpu = output.detach().cpu().squeeze(0).squeeze(0).numpy()
            plt.imshow(output_cpu, cmap='jet',)

            plt.title(f'Output (sum: {output_cpu.sum():.0f})')
            plt.axis('off')
            plt.colorbar()

            plt.tight_layout()
            # Сохраняем вместо отображения
            plt.savefig(f'./pic_train/train_epoch_{epoch}_batch_{i}.png', dpi=150, bbox_inches='tight')
            plt.close()

        
        
        loss = criterion(output, target)
        
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
    
def validate(val_list, model, criterion, device):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=args.batch_size)    
    
    model.eval()
    
    mae = 0

    mae_list = []
    
    for i,(img, target) in enumerate(test_loader):
        img = img.to(device)
        img = Variable(img)
        output = model(img)
        
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).to(device))

        mae_list.append((int(target.sum().item()), int(output.data.sum().item())))
        
    mae = mae/len(test_loader)    
    print(' * MAE {mae:.3f} '.format(mae=mae))
    print(mae_list)
    return mae    
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    
    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        
        scale = args.scales[i] if i < len(args.scales) else 1
        
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
if __name__ == '__main__':
    main()        
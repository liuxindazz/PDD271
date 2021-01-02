import os
import torch.utils.data
from torchvision import transforms
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config import BATCH_SIZE, LR, WD, INPUT_SIZE, Epoch
from core import model_densenet as model
# from core import model_resnet as model
#from core import model_vgg as model
from core import data_loader
from PIL import Image
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3')

train_transform = transforms.Compose([
    transforms.Resize((256, 256), Image.BILINEAR),
    transforms.RandomCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
    ])
test_transform = transforms.Compose([
    transforms.Resize((256, 256), Image.BILINEAR),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
    ])
trainset = data_loader.GetLoader(data_root='/home/LAB/wangzz/lxd/data/image_resize_224/', 
                    data_list='train_list.txt',  
                    transform=train_transform)
# print(len(trainset))
# print(len(trainset.labels))
# for data in trainset:
#     print(data[0].size(), data[1])
testset = data_loader.GetLoader(data_root='/home/LAB/wangzz/lxd/data/image_resize_224/', 
                data_list='validate_list.txt',  
                transform=test_transform)
# print(len(testset))
# print(len(testset.labels))
# for data in testset:
#     print(data[0].size(), data[1])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=4, drop_last=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=4, drop_last=False)
# define model
net = model.model_fusion()

criterion = torch.nn.CrossEntropyLoss()

# define optimizers
parameters = list(net.parameters())


optimizer = torch.optim.SGD(parameters, lr=LR, momentum=0.9, weight_decay=WD)

net = net.cuda()
net = DataParallel(net)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 500 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 500 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')
def save(net):
    torch.save(net.state_dict(), './models/resnet152.pth')
    print('Checkpoint saved to /models/resnet152.pth')

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


# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = LR * (0.1 ** (epoch // 30))
#     param_groups = optimizer.state_dict()['param_groups']
#     print (param_groups)
#     param_groups[0]['lr']=lr
#     param_groups[1]['lr']=lr*10
#     # for param_group in param_groups:
#     #     print param_group
#         # param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

best_prec1 = 0
for epoch in range(0, Epoch):
    # adjust_learning_rate(optimizer, epoch)

    # train for one epoch
    train(trainloader, net, criterion, optimizer, epoch)

    # evaluate on validation set
    prec1 = validate(testloader,net, criterion)
    # prec1 = test(epoch)
    # remember best prec@1 and save checkpoint
    is_best = prec1 > best_prec1
    if prec1 > best_prec1:
        save(net)


    best_prec1 = max(prec1, best_prec1)
print('finishing training')

import math
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils.ResNet import ResNet32
from utils.ResNet import ResNet18
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from utils.cutout import Cutout
from utils import cifar10Imbanlance
from utils import cifar100Imbanlance
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler

parser = argparse.ArgumentParser('arguments for training')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
parser.add_argument('--valid_size', type=int, default=0.2, help='valid_size')
parser.add_argument('--rate1', type=int, default=0.6, help='valid_size')
parser.add_argument('--rate2', type=int, default=0.4, help='valid_size')
parser.add_argument('--epochs', type=int, default=600, help='number of training epochs')
parser.add_argument('--eval_freq', default=1, type=int, help='evaluate model frequency')
# parser.add_argument('--save_freq', default=50, type=int, help='save model frequency')
parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--pin_memory', default=False, type=bool, help='GPU id to use.')
parser.add_argument('--dataset', type=str, default='cifar100', help="cifar10,cifar100,ImageNet-LT,iNaturelist2018")
parser.add_argument('--ir', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--pic_path', type=str, default='data/', help='path to dataset directory')
parser.add_argument('--num_cls', default=100, type=int, metavar='N',
                    help='number of classes in dataset (output dimention of models)')
parser.add_argument('--lr', '--learning_rate', default=0.10, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')

args = parser.parse_args()

# set device

gpu = 0
device = 'cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu'

# device = 'cuda:' if torch.cuda.is_available() else 'cpu'

# 读数据
# batch_size = 128
# percentage of training set to use as validation
# valid_size = 0.2
# ir = 0.01
# pic_path = 'data/'
# number of subprocesses to use for data loading
# num_workers = 4

"""
    batch_size: Number of loaded drawings per batch
    valid_size: Percentage of training set to use as validation
    num_workers: Number of subprocesses to use for data loading
    pic_path: The path of the pictrues
"""
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # R,G,B每层的归一化用到的均值和方差
    Cutout(n_holes=1, length=16),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 将数据转换为torch.FloatTensor，并标准化。

if args.dataset == 'cifar10':
    train_data = cifar10Imbanlance.Cifar10Imbanlance(transform=transform_train,
                                                     imbanlance_rate=args.ir, train=True,
                                                     file_path=args.pic_path)

    valid_data = cifar10Imbanlance.Cifar10Imbanlance(imbanlance_rate=args.ir, train=False,
                                                     transform=transform_test, file_path=args.pic_path)

    test_data = cifar10Imbanlance.Cifar10Imbanlance(imbanlance_rate=args.ir, train=False,
                                                    transform=transform_test, file_path=args.pic_path)

if args.dataset == 'cifar100':
    train_data = cifar100Imbanlance.Cifar100Imbanlance(transform=transform_train,
                                                       imbanlance_rate=args.ir, train=True,
                                                       file_path=os.path.join(args.pic_path, 'cifar-100-python/'))

    valid_data = cifar100Imbanlance.Cifar100Imbanlance(imbanlance_rate=args.ir, train=False,
                                                       transform=transform_test,
                                                       file_path=os.path.join(args.pic_path, 'cifar-100-python/'))

    test_data = cifar100Imbanlance.Cifar100Imbanlance(imbanlance_rate=args.ir, train=False,
                                                      transform=transform_test,
                                                      file_path=os.path.join(args.pic_path, 'cifar-100-python/'))

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
# random indices
np.random.shuffle(indices)
# the ratio of split
split = int(np.floor(args.valid_size * num_train))
# divide data to radin_data and valid_data
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
# 无放回地按照给定的索引列表采样样本元素
# train_sampler = SubsetRandomSampler(train_idx)
# valid_sampler = SubsetRandomSampler(valid_idx)


train_sampler = None
# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                           shuffle=(train_sampler is None), num_workers=args.num_workers,
                                           pin_memory=args.pin_memory, persistent_workers=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size,
                                           shuffle=False, num_workers=args.num_workers,
                                           pin_memory=args.pin_memory, persistent_workers=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers, pin_memory=args.pin_memory,
                                          persistent_workers=True)

if __name__ == "__main__":
    model = ResNet32()
    """
    ResNet18网络的7x7降采样卷积和池化操作容易丢失一部分信息,
    所以在实验中我们将7x7的降采样层和最大池化层去掉,替换为一个3x3的降采样卷积,
    同时减小该卷积层的步长和填充大小
    """
    # model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    # model.fc = torch.nn.Linear(512, args.num_cls)  # 将最后的全连接层改掉
    model = model.to(device)
    # 使用交叉熵损失函数
    criterion1 = nn.CrossEntropyLoss().to(device)
    criterion2 = nn.MSELoss().to(device)
    writer = SummaryWriter("./logs")

    # 开始训练
    valid_loss_min = np.Inf  # track change in validation loss
    accuracy = []
    lr = args.lr


    def adjust_learning_rate(optimizer, epoch, args):
        """Decay the learning rate based on schedule"""
        lr = args.lr
        # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    ac_max = 0
    for epoch in tqdm(range(1, args.epochs + 1)):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        total_sample = 0
        right_sample = 0

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        # 动态调整学习率
        adjust_learning_rate(optimizer, epoch, args)
        ###################
        # 训练集的模型 #
        ###################
        model.train()  # 作用是启用batch normalization和drop out
        for data, target in train_loader:
            # for i, (inputs, targets) in enumerate(train_loader):
            u = data

            data = data.to(device)
            target = target.to(device)
            # clear the gradients of all optimized variables（清除梯度）
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            # (正向传递：通过向模型传递输入来计算预测输出)
            output, rebuild = model(data)  # （等价于output = model.forward(data).to(device) ）
            # calculate the batch loss（计算损失值）
            loss = args.rate1 * criterion1(output, target) + args.rate2 * criterion2(rebuild, data)

            lo = criterion1(output, target)
            loo = criterion2(rebuild, data)
            # backward pass: compute gradient of the loss with respect to model parameters
            # （反向传递：计算损失相对于模型参数的梯度）
            loss.backward()
            # perform a single optimization step (parameter update)
            # 执行单个优化步骤（参数更新）
            optimizer.step()
            # update training loss（更新损失）
            train_loss += loss.item() * data.size(0)

        ######################
        # 验证集的模型#
        ######################
        if epoch % args.eval_freq == 0:
            model.eval()  # 验证模型
            for data, target in valid_loader:
                data = data.to(device)
                target = target.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output, rebuild = model(data)
                # calculate the batch loss
                loss = args.rate1 * criterion1(output, target) + args.rate2 * criterion2(rebuild, data)
                # update average validation loss
                valid_loss += loss.item() * data.size(0)
                # convert output probabilities to predicted class(将输出概率转换为预测类)
                _, pred = torch.max(output, 1)
                # compare predictions to true label(将预测与真实标签进行比较)
                correct_tensor = pred.eq(target.data.view_as(pred))
                # correct = np.squeeze(correct_tensor.to(device).numpy())
                total_sample += args.batch_size
                for i in correct_tensor:
                    if i:
                        right_sample += 1
            print("Accuracy:", 100 * right_sample / total_sample, "%")
            ac = right_sample / total_sample
            accuracy.append(ac)

            writer.add_scalar("Accuracy/val", ac, epoch)
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", valid_loss, epoch)
            writer.add_scalar("Learning rate", optimizer.param_groups[0]["lr"], epoch)

            # 显示训练集与验证集的损失函数
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))

            # 如果验证集损失函数减少，就保存模型。
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                valid_loss))
                torch.save(model.state_dict(), 'checkpoint/31006333v.pt')
                valid_loss_min = valid_loss
            if ac >= ac_max:
                print('ac decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(ac_max, ac))
                torch.save(model.state_dict(), 'checkpoint/310067559.pt')
                ac_max = ac

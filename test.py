import math
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torchvision
from matplotlib.pyplot import imshow
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from utils.cutout import Cutout
from utils import cifar10Imbanlance
from utils import cifar100Imbanlance
from utils.ResNet import ResNet18

parser = argparse.ArgumentParser('arguments for training')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--valid_size', type=int, default=0.2, help='valid_size')
parser.add_argument('--epochs', type=int, default=800, help='number of training epochs')
parser.add_argument('--eval_freq', default=20, type=int, help='evaluate model frequency')
# parser.add_argument('--save_freq', default=50, type=int, help='save model frequency')
parser.add_argument('--num_workers', type=int, default=2, help='num of workers to use')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--dataset', type=str, default='cifar10', help="cifar10,cifar100,ImageNet-LT,iNaturelist2018")
parser.add_argument('--ir', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--pic_path', type=str, default='data/', help='path to dataset directory')
parser.add_argument('--num_cls', default=10, type=int, metavar='N',
                    help='number of classes in dataset (output dimention of models)')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')

args = parser.parse_args()

gpu = 0
device = 'cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu'
# 读数据
# batch_size = 128
# percentage of training set to use as validation
# valid_size = 0.2
# ir = 0.01
# pic_path = 'data/'
# number of subprocesses to use for data loading
# num_workers = 4

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
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_sampler = None
# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                           shuffle=(train_sampler is None), num_workers=args.num_workers,
                                           pin_memory=False, persistent_workers=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size,
                                           shuffle=False, num_workers=args.num_workers,
                                           pin_memory=False, persistent_workers=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers, pin_memory=False, persistent_workers=True)

num_classes = len(np.unique(test_data.targets))
cls_num_list = [0] * num_classes
for label in train_data.targets:
    cls_num_list[label] += 1
test_cls_num_list = np.array(cls_num_list)
cls_num_list_cuda = torch.from_numpy(np.array(cls_num_list)).float().cuda()
all_preds = []
all_targets = []

if __name__ == "__main__":
    model = ResNet18()  # 得到预训练模型
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = torch.nn.Linear(512, args.num_cls)  # 将最后的全连接层修改
    # 载入权重

    # model.load_state_dict(torch.load('checkpoint/1257559.pt', map_location={'cuda:1':'cuda:0'}))

    #model.load_state_dict(torch.load('checkpoint/1257605.pt'))
    model.load_state_dict(torch.load('checkpoint/1257605.pt'))

    model = model.to(device)

    total_sample = 0
    right_sample = 0
    model.eval()  # 验证模型
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output, rebuild = model(data)

        # convert output probabilities to predicted class(将输出概率转换为预测类)
        _, pred = torch.max(output, 1)

        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        # compare predictions to true label(将预测与真实标签进行比较)
        correct_tensor = pred.eq(target.data.view_as(pred))
        # correct = np.squeeze(correct_tensor.to(device).numpy())
        total_sample += args.batch_size
        for i in correct_tensor:
            if i:
                right_sample += 1

    cf = confusion_matrix(all_targets, all_preds).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt
    many_shot = test_cls_num_list > 2000
    medium_shot = (test_cls_num_list <= 1000) & (test_cls_num_list > 300)
    few_shot = test_cls_num_list <= 100
    print("many avg, med avg, few avg",
          float(sum(cls_acc[many_shot]) * 100 / (sum(many_shot))),
          float(sum(cls_acc[medium_shot]) * 100 / (sum(medium_shot))),
          float(sum(cls_acc[few_shot]) * 100 / (sum(few_shot)))
          )
    print("Accuracy:", 100 * right_sample / total_sample, "%")

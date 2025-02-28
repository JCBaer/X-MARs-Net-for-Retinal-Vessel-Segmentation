import numpy as np
import torch.backends.cudnn as cudnn
import sys
sys.path.insert(0, '../')
import random
import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from six.moves import configparser
from XMARs_preprocess.image_handle import *
from XMARs_preprocess.extract_CHASEDB1 import get_data_training
from loss import *
from XMARs_Net import XMARs_Net

config = configparser.RawConfigParser()
config.read('../hyper-parameter_CHASEDB.txt')

path_data = config.get('data paths', 'path_local')
name_experiment = config.get('experiment name', 'name')
batch_size = int(config.get('training settings', 'batch_size'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_epoch = 0
total_epoch = 100
portion = 0.1
layers = 4
filters = 32
version = 2
best_loss = np.inf

lr_epoch = np.array([30,total_epoch])
lr_value= np.array([0.001,0.0001])

XMARs = XMARs_Net(classes=2,layers=layers,filters=filters,in_planes=1)

check_path = 'X-MARs-Net-v%d.pt7'%version

resume = False
loss_compute = LossMulti(jaccard_weight=0)
optimizer = optim.Adam(XMARs.parameters(),lr=lr_value[0])

images_train, training_masks = get_data_training(
    CHASEDB1_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
    CHASEDB1_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),
    patch_height = int(config.get('data attributes', 'patch_height')),
    patch_width = int(config.get('data attributes', 'patch_width')),
    N_subimgs = int(config.get('training settings', 'N_subimgs')),
    inside_FOV = config.getboolean('training settings', 'inside_FOV')
)

class TrainDataset(Dataset):
    def __init__(self, patches_imgs,training_masks):
        self.imgs = patches_imgs
        self.masks = training_masks

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        temp = self.masks[idx]
        temp = np.squeeze(temp,0)
        return torch.from_numpy(self.imgs[idx,...]).float(), torch.from_numpy(temp).long()

index = random.sample(range(training_masks.shape[0]),int(np.floor(portion*training_masks.shape[0])))

train_index =  set(range(training_masks.shape[0])) - set(index)
train_index = list(train_index)

print(f"Max index of train_index is: {max(train_index)}")

trainingset = TrainDataset(images_train[train_index,...],training_masks[train_index,...])
loader = DataLoader(trainingset, batch_size=batch_size,shuffle=True, num_workers=0)

newset = TrainDataset(images_train[index,...],training_masks[index,...])
new_loader = DataLoader(newset, batch_size=batch_size,shuffle=True, num_workers=0)

N = min(images_train.shape[0],40)
visualize(group_images(images_train[0:N,:,:,:],5),'../'+name_experiment+'/'+"imgs")
visualize(group_images(training_masks[0:N,:,:,:],5),'../'+name_experiment+'/'+"masks")

lr_schedule = np.zeros(total_epoch)
for l in range(len(lr_epoch)):
    if l ==0:
        lr_schedule[0:lr_epoch[l]] = lr_value[l]
    else:
        lr_schedule[lr_epoch[l-1]:lr_epoch[l]] = lr_value[l]

if device == 'cuda':
    XMARs.cuda()
    XMARs = torch.nn.DataParallel(XMARs, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

if resume:
    print('Waiting for resume: ')
    assert os.path.isdir('checkpoint'), 'No checkpoint found!'
    checkpoint = torch.load('./checkpoint/'+check_path)
    XMARs.load_state_dict(checkpoint['XMARs'])
    start_epoch = checkpoint['epoch']

def train(epoch):
    print('\nEpoch: %d' % epoch)
    XMARs.train()
    train_loss = 0
    lr = lr_schedule[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("Learning rate = %4f\n" % lr)

    for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = XMARs(inputs)
        loss = loss_compute(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print("Epoch %d: Train loss %6f\n" % (epoch, train_loss / np.float32(len(loader))))

def update(epoch):
    global best_loss
    XMARs.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(new_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = XMARs(inputs)
            loss = loss_compute(outputs, targets)

            test_loss += loss.item()

        print('Valid loss: {:.4f}'.format(test_loss))
    if test_loss < best_loss:
        print('Save to checkpoint.')
        record = {
            'XMARs': XMARs.state_dict(),
            'best_loss': best_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(record, './checkpoint/' + check_path)
        best_loss = test_loss

for epoch in range(start_epoch,total_epoch):
    train(epoch)
    update(epoch)
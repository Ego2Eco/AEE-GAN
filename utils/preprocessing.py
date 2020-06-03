import os
import math
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


class Scale(object):
    '''
    Given max and min of a rectangle it computes the scale and shift values to normalize data to [0,1]
    '''

    def __init__(self):
        self.min_x = +math.inf
        self.max_x = -math.inf
        self.min_y = +math.inf
        self.max_y = -math.inf
        self.sx, self.sy = 1, 1

    def calc_scale(self, keep_ratio=True):
        self.sx = 1 / (self.max_x - self.min_x)
        self.sy = 1 / (self.max_y - self.min_y)
        if keep_ratio:
            if self.sx > self.sy:
                self.sx = self.sy
            else:
                self.sy = self.sx

    def normalize(self, data, shift=True, inPlace=True):
        if inPlace:
            data_copy = data
        else:
            data_copy = np.copy(data)

        if data.ndim == 1:
            data_copy[0] = (data[0] - self.min_x * shift) * self.sx
            data_copy[1] = (data[1] - self.min_y * shift) * self.sy
        elif data.ndim == 2:
            data_copy[:, 0] = (data[:, 0] - self.min_x * shift) * self.sx
            data_copy[:, 1] = (data[:, 1] - self.min_y * shift) * self.sy
        elif data.ndim == 3:
            data_copy[:, :, 0] = (data[:, :, 0] - self.min_x * shift) * self.sx
            data_copy[:, :, 1] = (data[:, :, 1] - self.min_y * shift) * self.sy
        elif data.ndim == 4:
            data_copy[:, :, :, 0] = (data[:, :, :, 0] - self.min_x * shift) * self.sx
            data_copy[:, :, :, 1] = (data[:, :, :, 1] - self.min_y * shift) * self.sy
        else:
            return False
        return data_copy

    def denormalize(self, data, shift=True, inPlace=False):
        if inPlace:
            data_copy = data
        else:
            data_copy = np.copy(data)

        ndim = data.ndim
        if ndim == 1:
            data_copy[0] = data[0] / self.sx + self.min_x * shift
            data_copy[1] = data[1] / self.sy + self.min_y * shift
        elif ndim == 2:
            data_copy[:, 0] = data[:, 0] / self.sx + self.min_x * shift
            data_copy[:, 1] = data[:, 1] / self.sy + self.min_y * shift
        elif ndim == 3:
            data_copy[:, :, 0] = data[:, :, 0] / self.sx + self.min_x * shift
            data_copy[:, :, 1] = data[:, :, 1] / self.sy + self.min_y * shift
        elif ndim == 4:
            data_copy[:, :, :, 0] = data[:, :, :, 0] / self.sx + self.min_x * shift
            data_copy[:, :, :, 1] = data[:, :, :, 1] / self.sy + self.min_y * shift
        else:
            return False

        return data_copy



# Augment tensors of positions into positions+velocity
def get_traj_5d(obsv_p, pred_p):
    #print(obsv_p.shape)
    obsv_t = obsv_p[:,:,2]
    obsv_p = obsv_p[:,:,:2]
    obsv_v = obsv_p[:, 1:] - obsv_p[:, :-1]
    obsv_v = torch.cat([obsv_v[:, 0].unsqueeze(1), obsv_v], dim=1)
    #obsv_new = torch.randn(obsv_p.shape[0],obsv_p.shape[1]).cuda()
    obsv_5d = torch.cat([obsv_p, obsv_v, obsv_t.unsqueeze(2)],dim=2)
    #print(obsv_p.shape)
    obsv_4d = torch.cat([obsv_p, obsv_v], dim=2)
    #print(obsv_4d.shape)
    if len(pred_p) == 0: return obsv_5d
    pred_t = pred_p[:,:,2]
    pred_p = pred_p[:,:,:2]
    pred_p_1 = torch.cat([obsv_p[:, -1].unsqueeze(1), pred_p[:, :-1]], dim=1)
    pred_v = pred_p - pred_p_1
    #pred_new = torch.randn(pred_p.shape[0],pred_p.shape[1]).cuda()
    pred_5d = torch.cat([pred_p, pred_v, pred_t.unsqueeze(2)],dim=2)
    pred_4d = torch.cat([pred_p, pred_v], dim=2)
    return obsv_5d, pred_5d



def load_image(image_dir, dataset_t, device):
    loader = transforms.Compose([transforms.ToTensor(),])

    dataset_img = []
    images_label = np.unique(dataset_t)
    for i, label in enumerate(tqdm(images_label)):
        name = str(int(label))
        if 'eth' in image_dir:
            name = name + '.0'
        name = name + '.png'
        path = os.path.join(image_dir, name)
        img = Image.open(path).convert('RGB')
        img = img.resize((150,100), Image.ANTIALIAS)
        img_tensor = loader(img).unsqueeze(0)

        img_tensor = img_tensor.to(device)
        dataset_img.append(img_tensor)
    
    return torch.cat(dataset_img)


def merge_data(data, isval):
    dataset_obsv = []
    dataset_pred = []
    dataset_t = []
    the_batches = []
    dataset_img = []
    separate_point = []

    samples_accu = 0
    for i in range(len(data)):
        if isval:
            data[i]['the_batches'] -= data[i]['the_batches'][0][0]

        dataset_obsv.append(data[i]['dataset_obsv'])
        dataset_pred.append(data[i]['dataset_pred'])
        dataset_t.append(data[i]['dataset_t'])
        dataset_img.append(data[i]['dataset_img'])
        the_batches.append(data[i]['the_batches'] + samples_accu)
        samples_accu += data[i]['the_batches'][-1][1]
        separate_point.append(data[i]['the_batches'][-1][1])

    dataset_obsv = np.concatenate(dataset_obsv)
    dataset_pred = np.concatenate(dataset_pred)
    dataset_t = np.concatenate(dataset_t)
    the_batches = np.concatenate(the_batches)
    dataset_img = torch.cat(dataset_img)

    data = {'dataset_obsv': dataset_obsv, \
            'dataset_pred': dataset_pred, \
            'dataset_t': dataset_t, \
            'the_batches': the_batches, \
            'dataset_img': dataset_img }
    
    return data, separate_point
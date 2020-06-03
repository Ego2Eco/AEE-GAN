import os
import sys
import time
import math
import torch
import multiprocessing
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm, trange
from utils.preprocessing import Scale, get_traj_5d
from models.ptsemseg.models import get_model
from models.ptsemseg.loader import get_loader
from models.ptsemseg.utils import convert_state_dict

cwd = os.getcwd()

#=============================== Feature Encoder ===============================
def produce_fcn_features(image_dir, dataset, dataset_t, device):
    # Load all physicial images and extract features by a VGG model which
    # pretrain on cityscapes segmentation problem. feature shape:[1, 512, 10, 11]

    # Load the pretrain VGG model
    with torch.no_grad():
        model_dict = {"arch": "fcn8s"}
        # print('args.dataset:',)
        model = get_model(model_dict, 19, version="cityscapes")
        model_path = os.path.join(cwd, "models/pretrained_vgg/fcn8s_cityscapes_best_model.pkl")
        state = torch.load(model_path, map_location=device)["model_state"]
        state = convert_state_dict(state)
        model.load_state_dict(state)
        model.to(device)
    # Load the images and caculate the features
    loader = transforms.Compose([transforms.ToTensor(),])
    
    images_label = np.unique(dataset_t)

    VGG_features = []
    for i, label in enumerate(tqdm(images_label)):
        if dataset == 'waymo':
            name = str(label).zfill(6) + '.jpg'
        elif dataset == 'sdd':
            name = str(label) + '.0.png'
        path = os.path.join(image_dir, name)
        img = Image.open(path).convert('RGB')
        img_tensor = loader(img).unsqueeze(0)
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            _, feature = model(img_tensor)
        VGG_features.append(feature.squeeze())
    
    return torch.stack(VGG_features)


def produce_leave1out_fcn_features(images, device, img_batch_size=16):
    # Load the pretrain VGG model
    VGG_features = []
    with torch.no_grad():
        model_dict = {"arch": "fcn8s"}
        # print('args.dataset:',)
        model = get_model(model_dict, 19, version="cityscapes")
        model_path = os.path.join(cwd, "models/pretrained_vgg/fcn8s_cityscapes_best_model.pkl")
        state = torch.load(model_path, map_location=device)["model_state"]
        state = convert_state_dict(state)
        model.load_state_dict(state)
        model.to(device)
        stop = math.floor(len(images) / img_batch_size)
        for i in tqdm(range(0, stop, 1)):
            _, features = model(images[i:i+img_batch_size])
            VGG_features.append(features)
        if  len(images) % img_batch_size != 0:
            _, features = model(images[stop*img_batch_size:])
            VGG_features.append(features)
    return torch.cat(VGG_features)


class EncoderLstm(nn.Module):
    def __init__(self, hidden_size, n_layers=1):
        super(EncoderLstm, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Linear(5, self.hidden_size)

        # The LSTM cell.
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=n_layers, batch_first=True)
        self.lstm_h = []


    def init_lstm(self, h, c):
        # Initialization of the LSTM: hidden state and cell state
        self.lstm_h = (h, c)


    def forward(self, obsv):
        bs = obsv.shape[0]    # Batch size
        obsv = self.embed(obsv)
        # Reshape and applies LSTM over a whole sequence or over one single step
        y, self.lstm_h = self.lstm(obsv.view(bs, -1, self.hidden_size), self.lstm_h)
        return y



def DCA_MTX(x_4d, D_4d):
    Dp = D_4d[:, :, :2]
    Dv = D_4d[:, :, 2:4]
    DOT_Dp_Dv = torch.mul(Dp[:,:,0], Dv[:,:,0]) + torch.mul(Dp[:,:,1], Dv[:,:,1])
    Dv_sq = torch.mul(Dv[:,:,0], Dv[:,:,0]) + torch.mul(Dv[:,:,1], Dv[:,:,1]) + 1E-6
    TTCA = -torch.div(DOT_Dp_Dv, Dv_sq)
    DCA = torch.zeros_like(Dp)
    DCA[:, :, 0] = Dp[:, :, 0] + TTCA * Dv[:, :, 0]
    DCA[:, :, 1] = Dp[:, :, 1] + TTCA * Dv[:, :, 1]
    DCA = torch.norm(DCA, dim=2)
    return DCA


def BearingMTX(x_4d, D_4d):
    Dp = D_4d[:, :, :2]  # NxNx2
    v = x_4d[:, 2:4].unsqueeze(1).repeat(1, x_4d.shape[0], 1)  # => NxNx2
    DOT_Dp_v = Dp[:, :, 0] * v[:, :, 0] + Dp[:, :, 1] * v[:, :, 1]
    COS_THETA = torch.div(DOT_Dp_v, torch.norm(Dp, dim=2) * torch.norm(v, dim=2) + 1E-6)
    return COS_THETA


def SocialFeatures(x):
    N = x.shape[0]  # x is NxTx4 tensor
    # print('x:',x[:,-1])
    x_ver_repeat = x[:, -1].unsqueeze(0).repeat(N, 1, 1)
    x_hor_repeat = x[:, -1].unsqueeze(1).repeat(1, N, 1)
    # print('x_ver_repeat:',x_ver_repeat)
    # print('x_hor_repeat:',x_hor_repeat)
    Dx_mat = x_hor_repeat - x_ver_repeat
    hor_mat = x_ver_repeat - x_hor_repeat
    l2_dist_MTX = Dx_mat[:, :, :2].norm(dim=2)
    bearings_MTX = BearingMTX(x[:, -1], Dx_mat)
    hor_bearings_MTX = BearingMTX(x[:, -1], hor_mat)
    dcas_MTX = DCA_MTX(x[:, -1], Dx_mat)
    sFeatures_MTX = torch.stack([l2_dist_MTX, bearings_MTX, dcas_MTX], dim=2)

    return sFeatures_MTX, hor_bearings_MTX  # directly return the Social Features Matrix


def produce_social_features(dataset_obsv_5d, the_batches, batch_size, data_size):
    print('Compute social features: distance, bearing angle, smallest distance would reach')

    social_features_by_batch = []
    horizon_bearing_angles_by_batch = []

    batch_size_accum = 0
    sub_batches = []
    for ii, batch_i in enumerate(tqdm(the_batches)):
        batch_size_accum += batch_i[1] - batch_i[0]
        sub_batches.append(batch_i)

        if ii >= data_size - 1 or \
                batch_size_accum + (the_batches[ii + 1][1] - the_batches[ii + 1][0]) > batch_size:

            s_feature, hor_bearings = SocialFeatures(dataset_obsv_5d[sub_batches[0][0]:sub_batches[-1][1]])
            social_features_by_batch.append(s_feature)
            horizon_bearing_angles_by_batch.append(hor_bearings)

            batch_size_accum = 0
            sub_batches = []

    return social_features_by_batch, horizon_bearing_angles_by_batch



#================================= RVAE module =================================
class PhysicalSelfAttention(nn.Module):
    def __init__(self, input_channel, SA_channel):
        super(PhysicalSelfAttention, self).__init__()

        self.fcn_last_layer = nn.Conv2d(512,256,3)
        self.relu = nn.ReLU()
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.input_c = input_channel
        self.c_bar = SA_channel
        self.f = nn.Conv2d(self.input_c, self.c_bar, (1,1))
        self.g = nn.Conv2d(self.input_c, self.c_bar, (1,1))
        self.h = nn.Conv2d(self.input_c, self.c_bar, (1,1))
        self.v = nn.Conv2d(self.c_bar, self.input_c, (1,1))
        

    def forward(self, fcn_features):
        bs = fcn_features.shape[0]
        x = self.fcn_last_layer(fcn_features)
        x = self.relu(x)
        x = x.view(bs, self.input_c, 1, -1)
        fx = self.f(x).squeeze(2)
        gx = self.g(x).squeeze(2)
        hx = self.h(x).squeeze(2)
        s = torch.bmm(fx.transpose(1,2), gx)
        beta = torch.softmax(s, dim=1)
        o = self.v(torch.bmm(hx, beta).view(bs, self.c_bar, 1, -1))
        y = self.gamma * o + x
        return y



class EmbedSAPhysicalFeatures(nn.Module):
    def __init__(self, hidden_size):
        super(EmbedSAPhysicalFeatures, self).__init__()
        
        self.hidden_size = hidden_size
        self.fc = nn.Linear(18432, hidden_size)
        self.relu = nn.ReLU()


    def forward(self, SA_features):
        x = SA_features
        x = x.view(x.size(0), -1) #flat
        x = self.fc(x)
        return x



class PhysicalAttention(nn.Module):
    def __init__(self, h_dim, f_dim):
        super(PhysicalAttention, self).__init__()
        self.f_dim = f_dim
        self.h_dim = h_dim
        self.W = nn.Linear(h_dim, f_dim, bias=True)


    def forward(self, v_f, dec_h):
        Wh = self.W(dec_h)
        S = torch.zeros_like(dec_h)

        if dec_h.dim() == 1:
            sigma_i = v_f * Wh
            attentions = torch.softmax(sigma_i, dim=0)
            S = v_f * attentions
            S = S.unsqueeze(0)
        else:
            for ii in range(len(dec_h)):
                sigma_i = v_f * Wh[ii]
                attentions = torch.softmax(sigma_i, dim=0)
                S[ii] = v_f * attentions
        return S



#================================== SE module ==================================
class EmbedSocialFeatures(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EmbedSocialFeatures, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc = nn.Sequential(nn.Linear(input_size, 32), nn.ReLU(),
                                nn.Linear(32, 64), nn.ReLU(),
                                nn.Linear(64, hidden_size) )


    def forward(self, ftr_list):
        embedded_features = self.fc(ftr_list)
        return embedded_features



class AttentionPooling(nn.Module):
    def __init__(self, h_dim, f_dim):
        super(AttentionPooling, self).__init__()
        self.f_dim = f_dim
        self.h_dim = h_dim
        self.W = nn.Linear(h_dim, f_dim, bias=True)


    def forward(self, f, h, sub_batches):
        Wh = self.W(h)
        S = torch.zeros_like(h)
        for sb in sub_batches:
            N = sb[1] - sb[0]
            if N == 1: continue

            for ii in range(sb[0], sb[1]):
                fi = f[ii, sb[0]:sb[1]]
                sigma_i = torch.bmm(fi.unsqueeze(1), Wh[sb[0]:sb[1]].unsqueeze(2))
                sigma_i[ii-sb[0]] = -1000

                attentions = torch.softmax(sigma_i.squeeze(), dim=0)
                S[ii] = torch.mm(attentions.view(1, N), h[sb[0]:sb[1]])
        if S.ndim == 1:
            S = S.unsqueeze(0)
        return S



class Horizon_AttentionPooling(nn.Module):
    def __init__(self, h_dim, f_dim):
        super(Horizon_AttentionPooling, self).__init__()
        self.f_dim = f_dim
        self.h_dim = h_dim
        self.W = nn.Linear(h_dim, f_dim, bias=True)


    def forward(self, f, h, sub_batches, features, hor_bearings_MTX):
        # f is social feature, h is lstm hidden state

        distance = features[:,:,0]
        angle = hor_bearings_MTX       
        Wh = self.W(h)
        
        S = torch.zeros_like(h)

        for sb in sub_batches:
            N = sb[1] - sb[0]
            if N == 1: continue
            for ii in range(sb[0], sb[1]):
                fi = f[ii, sb[0]:sb[1]]
                d = distance[ii, sb[0]:sb[1]]
                theta = angle[ii, sb[0]:sb[1]]

                sigma_i = torch.bmm(fi.unsqueeze(1), Wh[sb[0]:sb[1]].unsqueeze(2))
                sigma_i[ii-sb[0]] = -1000

                constraint = np.where((theta < 0)| (d > 10))
                sigma_i[constraint] = -1000

                if constraint[0].shape == sb[1] - sb[0] - 1:
                    attentions = torch.zeros_like(sigma_i.squeeze())
                else:
                    attentions = torch.softmax(sigma_i.squeeze(), dim=0)

                S[ii] = torch.mm(attentions.view(1, N), h[sb[0]:sb[1]]) # (1,N) mm (N,64) ; 64 is embedding size

        if S.ndim == 1:
            S = S.unsqueeze(0)
        return S




#=================================== Decoder ===================================
class DecoderLstm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DecoderLstm, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        # Fully connected sub-network. Input is hidden_size, output is 2.
        self.fc = nn.Sequential(torch.nn.Linear(hidden_size, 64), nn.Sigmoid(),
                                torch.nn.Linear(64, 64), nn.LeakyReLU(0.2),
                                torch.nn.Linear(64, 32), nn.LeakyReLU(0.2),
                                torch.nn.Linear(32, 2) )
        self.lstm_h = []


    def init_lstm(self, h, c):
        # Initialization of the LSTM: hidden state and cell state
        self.lstm_h = (h, c)


    def forward(self, h, s, v_s, z):
        bs = z.shape[0]    # Batch size
        # For each sample in the batch, concatenate h (hidden state), s (social term) and z (noise)
        inp = torch.cat([h, s, v_s, z], dim=1)
        # Applies a forward step.
        out, self.lstm_h = self.lstm(inp.unsqueeze(1), self.lstm_h)
        # Applies the fully connected layer to the LSTM output
        out = self.fc(out.squeeze())
        return out




#================================ Discriminator ================================
class Discriminator(nn.Module):
    def __init__(self, n_next, hidden_dim, n_latent_code, device):
        super(Discriminator, self).__init__()
        self.lstm_dim = hidden_dim
        self.n_next = n_next
        self.device = device

        self.obsv_encoder_lstm = nn.LSTM(5, hidden_dim, batch_first=True)
        self.obsv_encoder_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
                                             nn.Linear(hidden_dim // 2, hidden_dim // 2))

        self.pred_encoder = nn.Sequential(nn.Linear(n_next * 5, hidden_dim // 2), nn.LeakyReLU(0.2),
                                          nn.Linear(hidden_dim // 2, hidden_dim // 2))

        # Classifier: input is hidden_dim (concatenated encodings of observed and predicted trajectories), output is 1
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
                                        nn.Linear(hidden_dim // 2, 1))
        # Latent code inference: input is hidden_dim (concatenated encodings of observed and predicted trajectories), output is n_latent_code (distribution of latent codes)
        self.latent_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
                                            nn.Linear(self.lstm_dim // 2, n_latent_code))


    def forward(self, obsv, pred):
        obsv = obsv[:,:,:5].contiguous()
        pred = pred[:,:,:5].contiguous()
        bs = obsv.size(0)
        lstm_h_c = (torch.zeros(1, bs, self.lstm_dim).to(self.device),
                    torch.zeros(1, bs, self.lstm_dim).to(self.device))
        # Encoding of the observed sequence trhough an LSTM cell
        obsv_code, lstm_h_c = self.obsv_encoder_lstm(obsv, lstm_h_c)
        # Further encoding through a FC layer
        obsv_code = self.obsv_encoder_fc(obsv_code[:, -1])
        # Encoding of the predicted/next part of the sequence through a FC layer
        pred_code = self.pred_encoder(pred.view(-1, self.n_next * 5))

        both_codes = torch.cat([obsv_code, pred_code], dim=1)
        # Applies classifier to the concatenation of the encodings of both parts
        label = self.classifier(both_codes)
        # Inference on the latent code
        code_hat = self.latent_decoder(both_codes)
        return label, code_hat


    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()


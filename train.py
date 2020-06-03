import os
import time
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as opt
from torchvision import transforms
from tqdm import tqdm, trange
from itertools import chain
from torch.autograd import Variable
from utils.preprocessing import Scale, get_traj_5d
from models.model_utils import FutureHiddenStateRegister
from models.train_test_func import train, evaluate, test, share_variables
from models.models import *


cwd = os.getcwd()
print('Current working directory: ', cwd)

# ========== Set input/output files ============
dataset = 'waymo'

dataset_dir = os.path.join(cwd, 'datasets')
if dataset == 'waymo':
    dataset_name = 'waymo0004'
    image_dir = os.path.join(dataset_dir, 'waymo0004/waymo0004_image')
    input_file = os.path.join(dataset_dir, 'waymo0004/waymo0004_6s_typeid.npz')
elif dataset == 'sdd':
    dataset_name = 'sdd_nexus6'
    image_dir = os.path.join(dataset_dir, 'sdd_nexus6/sdd_nexus6_image')
    input_file = os.path.join(dataset_dir, 'sdd_nexus6/sdd_nexus6_typeid.npz')


model_name = 'aeegan-v3'
model_file = ''
evaluate_result_dir = os.path.join('evaluate_result_dir', model_name + '_' + dataset_name)
model_checkpoints_dir = 'checkpoints/' + model_name + '_' + dataset_name


# ========== Training hyper-parameters =========
pretrain = False
use_gpu = True
num_of_gpus = 1  # 1 or 2
gpu_device = 0
use_parallel = False
use_visual = True
use_social = True
use_horizon = True
n_gen_samples = 20

# Info GAN
use_info_loss = True
loss_info_w = 0.5
n_latent_codes = 2
# L2 GAN
use_l2_loss = False
use_variety_loss = False
loss_l2_w = 0.5  # WARNING for both l2 and variety
# Learning Rate
lr_g = 5E-5
lr_d = 1E-4


# ============= Network Size ===================
batch_size = 256
hidden_size = 64
n_epochs = 10000
n_lstm_layers = 1
noise_len = hidden_size // 2
num_social_features = 3
social_feature_size = hidden_size
horizon_social_feature_size = hidden_size
visual_feature_size = hidden_size


# ============= device setting ===================
if not use_gpu:
    device1 = torch.device('cpu')
    device2 = torch.device('cpu')
    print('Training on cpu')
    if torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
else:
    if num_of_gpus == 1:
        device_name = 'cuda:' + str(gpu_device)
        device1 = torch.device(device_name)
        device2 = torch.device(device_name)
        print('Training on', device_name)
    else:
        device1 = torch.device('cuda:0')
        device2 = torch.device('cuda:1')
        print('Training on cuda:0 and cuda:1')


# ============== Loda Data =====================
data = np.load(input_file)
# Data come as NxTx2 numpy nd-arrays where N is the number of trajectories,
# T is their duration.
dataset_obsv, dataset_pred, dataset_t, the_batches = \
    data['obsvs'], data['preds'], data['times'], data['batches']

# 4/5 of the batches to be used for training phase, 1/5 for testing
# 7/8 of the batches in training phase to be used for training, 1/8 for validation
train_val_size = max(1, (len(the_batches) * 4) // 5)
train_size = max(1, (train_val_size * 7) // 8)
val_size = train_val_size - train_size
test_size = len(the_batches) - train_val_size

train_batches = the_batches[:train_size]
val_batches = the_batches[train_size:train_val_size]
test_batches = the_batches[train_val_size:]

n_past = dataset_obsv.shape[1]  # Size of the observed sub-paths
n_next = dataset_pred.shape[1]  # Size of the sub-paths to predict

n_total_samples = the_batches[-1][1]
n_train_samples = the_batches[train_size - 1][1]  # Number of training samples
print('Data path:', input_file)
print(' # Total samples:', n_total_samples, ' # Training samples:', n_train_samples)


# ================ Normalization ================
scale = Scale()
scale.max_x = max(np.max(dataset_obsv[:, :, 0]), np.max(dataset_pred[:, :, 0]))
scale.min_x = min(np.min(dataset_obsv[:, :, 0]), np.min(dataset_pred[:, :, 0]))
scale.max_y = max(np.max(dataset_obsv[:, :, 1]), np.max(dataset_pred[:, :, 1]))
scale.min_y = min(np.min(dataset_obsv[:, :, 1]), np.min(dataset_pred[:, :, 1]))
scale.calc_scale(keep_ratio=True)
dataset_obsv = scale.normalize(dataset_obsv)
dataset_pred = scale.normalize(dataset_pred)
ss = scale.sx
# Copy normalized observations/paths to predict into torch GPU tensors
dataset_obsv = torch.FloatTensor(dataset_obsv)
dataset_pred = torch.FloatTensor(dataset_pred)


# =================== Create social features and vidual features ====================
dataset_agent_ids = dataset_obsv[:, 0, 3].tolist()
dataset_obsv_5d, dataset_pred_5d = get_traj_5d(dataset_obsv, dataset_pred)

dataset_obsv_5d = dataset_obsv_5d.to(device1)
dataset_pred_5d = dataset_pred_5d.to(device1)

train_social_features, train_horizon_angles = produce_social_features(dataset_obsv_5d, train_batches, batch_size, train_size)
val_social_features, val_horizon_angles = produce_social_features(dataset_obsv_5d, val_batches, batch_size, val_size)
test_social_features, test_horizon_angles = produce_social_features(dataset_obsv_5d, test_batches, batch_size, test_size)

all_fcn_features = produce_fcn_features(image_dir, dataset, dataset_t, device1)


# ==================== Create model module object =====================
picture_sa_embedder = EmbedSAPhysicalFeatures(hidden_size).to(device1)

encoder = EncoderLstm(hidden_size, n_lstm_layers).to(device1)
feature_embedder = EmbedSocialFeatures(num_social_features, social_feature_size).to(device1)
attention = AttentionPooling(hidden_size, social_feature_size).to(device1)
v_attention = PhysicalAttention(hidden_size, social_feature_size).to(device1)
v_self_attention = PhysicalSelfAttention(256, 64).to(device1)
hor_attention = Horizon_AttentionPooling(hidden_size, horizon_social_feature_size).to(device1)
dec_h_register = FutureHiddenStateRegister(hidden_size, n_lstm_layers)
decoder = DecoderLstm(hidden_size + social_feature_size + visual_feature_size + noise_len, hidden_size).to(device1)

# The Generator parameters and their optimizer
predictor_params = chain(feature_embedder.parameters(), attention.parameters(), hor_attention.parameters(), encoder.parameters(), 
                         v_attention.parameters(), decoder.parameters(), picture_sa_embedder.parameters(),
                         v_self_attention.parameters())
predictor_optimizer = opt.Adam(predictor_params, lr=lr_g, betas=(0.9, 0.999))

# The Discriminator parameters and their optimizer
D = Discriminator(n_next, hidden_size, n_latent_codes, device1).to(device1)
D_optimizer = opt.Adam(D.parameters(), lr=lr_d, betas=(0.9, 0.999))

mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()



# Share the global variables to the module 'train_test_func.py'
#     Note that the value type variables (ex. int, str class) cannot overwrite in other module, 
#     because this function only copy the dict to update the global variables of train_test_func.py.
#     When you modify these variables, it just changes the address which copyed variable reference.
global_variables = globals()
share_variables(**global_variables)


# =======================================================
# ===================== M A I N =========================
# =======================================================
if pretrain == True:
    if os.path.isfile(model_file):
        print('Loading model from ' + model_file)
        checkpoint = torch.load(model_file)
    else:
        checkpoints = os.listdir(model_checkpoints_dir)
        assert len(checkpoints) != 0, 'There is no pretrain checkpoint.'
        checkpoints.sort(key=lambda name: int(name.split('_')[-1][0:-3]))
        last_checkpoint = os.path.join(model_checkpoints_dir, checkpoints[-1])
        print('Loading model from ' + last_checkpoint)
        checkpoint = torch.load(last_checkpoint)
    
    start_epoch = checkpoint['epoch'] + 1
    attention.load_state_dict(checkpoint['attentioner_dict'])
    hor_attention.load_state_dict(checkpoint['hor_attention_dict'])
    encoder.load_state_dict(checkpoint['lstm_encoder_dict'])
    decoder.load_state_dict(checkpoint['predictor_dict'])
    predictor_optimizer.load_state_dict(checkpoint['pred_optimizer'])
    feature_embedder.load_state_dict(checkpoint['feature_embedder_dict'])
    picture_sa_embedder.load_state_dict(checkpoint['picture_sa_embedder'])
    v_attention.load_state_dict(checkpoint['v_attention_dict'])
    v_self_attention.load_state_dict(checkpoint['v_self_attention'])
    D.load_state_dict(checkpoint['D_dict'])
    D_optimizer.load_state_dict(checkpoint['D_optimizer'])
else:
    min_train_ADE = 10000
    start_epoch = 1


if not os.path.exists('evaluate_result_dir'):
    os.mkdir('evaluate_result_dir')
if not os.path.exists(evaluate_result_dir):
    os.mkdir(evaluate_result_dir)

if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')
if not os.path.exists(model_checkpoints_dir):
    os.mkdir(model_checkpoints_dir)
    

# ===================== TRAIN =========================
for epoch in trange(start_epoch, n_epochs + 1):  # FIXME : set the number of epochs
    # Main training function
    train(epoch)

    # ============== Save model on disk ===============
    if epoch % 50 == 0:  # FIXME : set the interval for running tests
        model_file = model_checkpoints_dir + '/' + model_name + '_' + dataset_name + \
                     '_' + str(epoch) + '.pt'
        print('Saving model to file ...', model_file)
        torch.save({
            'epoch': epoch,
            'attentioner_dict': attention.state_dict(),
            'hor_attention_dict': hor_attention.state_dict(),
            'lstm_encoder_dict': encoder.state_dict(),
            'predictor_dict': decoder.state_dict(),
            'pred_optimizer': predictor_optimizer.state_dict(),
            'feature_embedder_dict': feature_embedder.state_dict(),
            'picture_sa_embedder': picture_sa_embedder.state_dict(),
            'v_attention_dict' : v_attention.state_dict(),
            'v_self_attention': v_self_attention.state_dict(),
            'D_dict': D.state_dict(),
            'D_optimizer': D_optimizer.state_dict(),
        }, model_file)
        wr_dir = 'medium/' + dataset_name + '/' + model_name + '/' + str(epoch)
        os.makedirs(wr_dir, exist_ok=True)
        evaluate(epoch)

test(get_best=True)

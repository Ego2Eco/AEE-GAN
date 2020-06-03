import os
import time
import copy
import torch
import numpy as np
from torch.autograd import Variable
from models.model_utils import ModelSelector


def share_variables(**kwargs):
    global_variables = globals()
    global_variables.update(kwargs)



def generator(noise, encoder, s_weighted_feature, v_weighted_feature, obsv_5d):
    pred_5d = []
    last_obsv = obsv_5d[:, -1]
    type_info = obsv_5d[:,-1,4]
    N = obsv_5d.shape[0]

    for i in range(n_next):
        new_v = decoder(encoder.lstm_h[0].view(N, -1), s_weighted_feature, v_weighted_feature, noise).view(N, 2)
        new_p = new_v + last_obsv[:, :2]
        last_obsv = torch.cat([new_p, new_v, type_info.unsqueeze(1)], dim=1)
        pred_5d.append(last_obsv)
        encoder(pred_5d[-1])

    return torch.stack(pred_5d, 1)



def train(epoch):
    tic = time.perf_counter()
    dec_h_register.reset()

    train_ADE, train_FDE = 0, 0
    batch_index = 0
    batch_size_accum = 0
    sub_batches = []
    frame_indexes = []
    # For all the training batches
    for ii, batch_i in enumerate(train_batches):
        batch_size_accum += batch_i[1] - batch_i[0]
        sub_batches.append(batch_i)
        frame_indexes.append(ii)

        if ii >= train_size - 1 or \
                batch_size_accum + (train_batches[ii + 1][1] - train_batches[ii + 1][0]) > batch_size:
            
            bs = batch_size_accum
            # Slice the batch portion from all data
            batch_ids = dataset_agent_ids[sub_batches[0][0]:sub_batches[-1][1]]
            batch_obsv_5d = dataset_obsv_5d[sub_batches[0][0]:sub_batches[-1][1]]
            batch_pred_5d = dataset_pred_5d[sub_batches[0][0]:sub_batches[-1][1]]
            batch_fcn_features = all_fcn_features[frame_indexes[0]:frame_indexes[-1]+1]
            batch_social_features = train_social_features[batch_index]
            batch_horizon_angles = train_horizon_angles[batch_index]
            sub_batches = sub_batches - sub_batches[0][0]
            
            lstm_h_c = (torch.zeros(n_lstm_layers, bs, encoder.hidden_size).to(device1),
                        torch.zeros(n_lstm_layers, bs, encoder.hidden_size).to(device1))
            encoder.init_lstm(lstm_h_c[0], lstm_h_c[1])

            encoder(batch_obsv_5d)
            batch_encoder_hidden_state = encoder.lstm_h[0]
            batch_encoder_cell_state = encoder.lstm_h[1]

            if use_visual:
                batch_v_feature = v_self_attention(batch_fcn_features)
                batch_v_feature = picture_sa_embedder(batch_v_feature)

            if use_social or use_horizon:
                emb_features = feature_embedder(batch_social_features)
                weighted_features = torch.zeros_like(encoder.lstm_h[0].squeeze())
                hor_weighted_features = torch.zeros_like(encoder.lstm_h[0].squeeze())
                if use_social:
                    weighted_features = attention(emb_features, encoder.lstm_h[0].squeeze(), sub_batches)
                if use_horizon:
                    hor_weighted_features = hor_attention(emb_features, encoder.lstm_h[0].squeeze(), sub_batches, \
                                                          batch_social_features.data.cpu().numpy(), batch_horizon_angles.data.cpu().numpy())
                batch_s_weighted_feature = weighted_features + hor_weighted_features
            else:
                batch_s_weighted_feature = torch.zeros_like(encoder.lstm_h[0].squeeze())
            

            for i, sb in enumerate(sub_batches):
                N = sb[1] - sb[0]
                zeros = Variable(torch.zeros(N, 1) + np.random.uniform(0, 0.1), requires_grad=False).to(device1)
                ones = Variable(torch.ones(N, 1) * np.random.uniform(0.9, 1.0), requires_grad=False).to(device1)
                noise = torch.FloatTensor(torch.rand(N, noise_len)).to(device1)

                # Slice the frame portion from batch data
                ids = batch_ids[sb[0]:sb[1]]
                obsv_5d = batch_obsv_5d[sb[0]:sb[1]]
                pred_5d = batch_pred_5d[sb[0]:sb[1]]
                s_weighted_feature = batch_s_weighted_feature[sb[0]:sb[1]]
                v_feature = batch_v_feature[i]

                encoder.lstm_h = (batch_encoder_hidden_state[:, sb[0]:sb[1], :],\
                                  batch_encoder_cell_state[:, sb[0]:sb[1], :]) 

                lstm_h_c = (torch.zeros(n_lstm_layers, N, decoder.hidden_size).to(device1),
                            torch.zeros(n_lstm_layers, N, decoder.hidden_size).to(device1))
                decoder.init_lstm(lstm_h_c[0], lstm_h_c[1])

                dec_h = dec_h_register.pop(ids, device1)
                v_weighted_feature = v_attention(v_feature, dec_h.squeeze())

                # ============== Train Discriminator ================
                D.zero_grad()
                pred_hat_5d = generator(noise, encoder, s_weighted_feature, v_weighted_feature, obsv_5d)
                pred_hat_5d_detach = pred_hat_5d.detach()

                fake_labels, code_hat = D(obsv_5d, pred_hat_5d_detach)
                a_shape = code_hat.squeeze().shape
                # Evaluate the MSE loss: the fake_labels should be close to zero
                d_loss_fake = mse_loss(fake_labels, zeros)
                ashape = code_hat.squeeze().shape
                bshape = noise[:, :n_latent_codes].shape
                d_loss_info = mse_loss(code_hat, noise[:, :n_latent_codes])
                # Evaluate the MSE loss: the real should be close to one
                real_labels, code_hat = D(obsv_5d, pred_5d)
                d_loss_real = mse_loss(real_labels, ones)

                d_loss = d_loss_fake + d_loss_real
                if use_info_loss:
                    d_loss += loss_info_w * d_loss_info
                
                d_loss.backward(retain_graph=True)  # update D
                D_optimizer.step()

                # =============== Train Generator ================= #
                # Zero the gradient buffers of all the discriminator parameters
                D.zero_grad()
                # Zero the gradient buffers of all the generator parameters
                predictor_optimizer.zero_grad()

                # Classify the generated fake sample
                gen_labels, code_hat = D(obsv_5d, pred_hat_5d)
                
                # Adversarial loss (classification labels should be close to one)
                g_loss_fooling = mse_loss(gen_labels, ones)
                g_loss = g_loss_fooling

                if use_info_loss:
                    # Information loss
                    g_loss_info = mse_loss(code_hat, noise[:, :n_latent_codes])
                    g_loss += loss_info_w * g_loss_info
                if use_l2_loss:
                    # L2 loss between the predicted paths and the true ones
                    g_loss_l2 = mse_loss(pred_hat_5d[:, :, :2], pred_5d[:,:,:2])
                    g_loss += loss_l2_w * g_loss_l2

                g_loss.backward(retain_graph=True)
                predictor_optimizer.step()

                dec_h_register.save(ids, decoder.lstm_h[0])

                # calculate error
                with torch.no_grad():  # TODO: use the function above
                    err_all = torch.pow((pred_hat_5d[:, :, :2] - pred_5d[:,:,:2]) / ss, 2)
                    #print(err_all.shape)
                    err_all = err_all.sum(dim=2).sqrt()
                    e = err_all.sum().item() / n_next
                    train_ADE += e
                    train_FDE += err_all[:, -1].sum().item()

                predictor_optimizer.zero_grad()


            batch_index = batch_index + 1
            batch_size_accum = 0
            sub_batches = []
            frame_indexes = []


    train_ADE /= n_train_samples
    train_FDE /= n_train_samples
    toc = time.perf_counter()
    print(" Epc=%4d, Train ADE,FDE = (%.3f, %.3f) | time = %.1f" \
        % (epoch, train_ADE, train_FDE, toc - tic))
    if (epoch%5 == 0):
        train_file_name = os.path.join(evaluate_result_dir, "train.txt")
        f_train = open(train_file_name,"a")
        lines = "Epc=%4d, Train ADE,FDE = (%.3f, %.3f) | time = %.1f\n" % (epoch, train_ADE, train_FDE, toc - tic)
        f_train.writelines(lines)
        f_train.close()

            


def evaluate(epoch):
    dec_h_register.reset()
    
    # =========== Test error ============
    ade_avg_12, fde_avg_12 = 0, 0
    ade_min_12, fde_min_12 = 0, 0
    val_samples = 0

    batch_index = 0
    batch_size_accum = 0
    sub_batches = []
    frame_indexes = []

    for ii, batch_i in enumerate(val_batches):
        batch_size_accum += batch_i[1] - batch_i[0]
        val_samples += int(batch_i[1]- batch_i[0])
        sub_batches.append(batch_i)
        frame_indexes.append(train_size + ii)

        if ii >= val_size - 1 or \
                batch_size_accum + (val_batches[ii + 1][1] - val_batches[ii + 1][0]) > batch_size:
            
            bs = batch_size_accum
            # Slice the batch portion from all data
            batch_ids = dataset_agent_ids[sub_batches[0][0]:sub_batches[-1][1]]
            batch_obsv_5d = dataset_obsv_5d[sub_batches[0][0]:sub_batches[-1][1]]
            batch_pred_5d = dataset_pred_5d[sub_batches[0][0]:sub_batches[-1][1]]
            batch_fcn_features = all_fcn_features[frame_indexes[0]:frame_indexes[-1]+1]
            batch_social_features = val_social_features[batch_index]
            batch_horizon_angles = val_horizon_angles[batch_index]
            sub_batches = sub_batches - sub_batches[0][0]
            
            lstm_h_c = (torch.zeros(n_lstm_layers, bs, encoder.hidden_size).to(device1),
                        torch.zeros(n_lstm_layers, bs, encoder.hidden_size).to(device1))
            encoder.init_lstm(lstm_h_c[0], lstm_h_c[1])

            with torch.no_grad():
                encoder(batch_obsv_5d)
                batch_encoder_hidden_state = encoder.lstm_h[0]
                batch_encoder_cell_state = encoder.lstm_h[1]

                if use_visual:
                    batch_v_feature = v_self_attention(batch_fcn_features)
                    batch_v_feature = picture_sa_embedder(batch_v_feature)

                if use_social or use_horizon:
                    emb_features = feature_embedder(batch_social_features)
                    weighted_features = torch.zeros_like(encoder.lstm_h[0].squeeze())
                    hor_weighted_features = torch.zeros_like(encoder.lstm_h[0].squeeze())
                    if use_social:
                        weighted_features = attention(emb_features, encoder.lstm_h[0].squeeze(), sub_batches)
                    if use_horizon:
                        hor_weighted_features = hor_attention(emb_features, encoder.lstm_h[0].squeeze(), sub_batches, \
                                                            batch_social_features.data.cpu().numpy(), batch_horizon_angles.data.cpu().numpy())
                    batch_s_weighted_feature = weighted_features + hor_weighted_features
                else:
                    batch_s_weighted_feature = torch.zeros_like(encoder.lstm_h[0].squeeze())


                for i, sb in enumerate(sub_batches):
                    N = sb[1] - sb[0]
                    zeros = Variable(torch.zeros(N, 1) + np.random.uniform(0, 0.1), requires_grad=False).to(device1)
                    ones = Variable(torch.ones(N, 1) * np.random.uniform(0.9, 1.0), requires_grad=False).to(device1)

                    # Slice the frame portion from batch data
                    ids = batch_ids[sb[0]:sb[1]]
                    obsv_5d = batch_obsv_5d[sb[0]:sb[1]]
                    pred_5d = batch_pred_5d[sb[0]:sb[1]]
                    s_weighted_feature = batch_s_weighted_feature[sb[0]:sb[1]]
                    v_feature = batch_v_feature[i]

                    encoder.lstm_h = (batch_encoder_hidden_state[:, sb[0]:sb[1], :],\
                                    batch_encoder_cell_state[:, sb[0]:sb[1], :]) 

                    lstm_h_c = (torch.zeros(n_lstm_layers, N, decoder.hidden_size).to(device1),
                                torch.zeros(n_lstm_layers, N, decoder.hidden_size).to(device1))
                    decoder.init_lstm(lstm_h_c[0], lstm_h_c[1])

                    dec_h = dec_h_register.pop(ids, device1)
                    v_weighted_feature = v_attention(v_feature, dec_h.squeeze())

                    
                    all_20_errors = []
                    all_20_preds = []

                    for kk in range(n_gen_samples):
                        noise = torch.FloatTensor(torch.rand(N, noise_len)).to(device1)
                        pred_hat_5d = generator(noise, encoder, s_weighted_feature, v_weighted_feature, obsv_5d)
                        if kk == n_gen_samples - 1:
                            dec_h_register.save(ids, decoder.lstm_h[0])

                        all_20_preds.append(pred_hat_5d.unsqueeze(0))
                        err_all = torch.pow((pred_hat_5d[:, :, :2] - pred_5d[:,:,:2]) / ss, 2).sum(dim=2, keepdim=True).sqrt()
                        all_20_errors.append(err_all.unsqueeze(0))
                    
                    all_20_errors = torch.cat(all_20_errors)

                    # =============== Prediction Errors ================
                    fde_min_12_i, _ = all_20_errors[:, :, -1].min(0, keepdim=True)
                    ade_min_12_i, _ = all_20_errors.mean(2).min(0, keepdim=True)
                    fde_min_12 += fde_min_12_i.sum().item()
                    ade_min_12 += ade_min_12_i.sum().item()
                    fde_avg_12 += all_20_errors[:, :, -1].mean(0, keepdim=True).sum().item()
                    ade_avg_12 += all_20_errors.mean(2).mean(0, keepdim=True).sum().item()
                    # ==================================================


            batch_index = batch_index + 1
            batch_size_accum = 0
            sub_batches = []
            frame_indexes = []


    print("validate samples:", val_samples)
    ade_avg_12 /= val_samples
    fde_avg_12 /= val_samples
    ade_min_12 /= val_samples
    fde_min_12 /= val_samples
    print('Avg ADE,FDE (12)= (%.3f, %.3f) | Min(20) ADE,FDE (12)= (%.3f, %.3f)' \
          % (ade_avg_12, fde_avg_12, ade_min_12, fde_min_12))


    val_file_name = os.path.join(evaluate_result_dir, "val.txt")
    f_val = open(val_file_name, "a")
    lines = 'Epoch(%d): Avg ADE,FDE (12)= (%.3f, %.3f) | Min(20) ADE,FDE (12)= (%.3f, %.3f)\n' \
          % (epoch, ade_avg_12, fde_avg_12, ade_min_12, fde_min_12)
    f_val.writelines(lines)
    f_val.close()




def test(epoch=None, get_best=False):
    if get_best == True:
        model_selector = ModelSelector()
        best_model_path = model_selector.get_model(model_name, dataset_name, evaluate_result_dir, model_checkpoints_dir)

        print('Loading model from ' + best_model_path)
        checkpoint = torch.load(best_model_path)
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



    dec_h_register.reset()
    
    # =========== Test error ============
    ade_avg_12, fde_avg_12 = 0, 0
    ade_min_12, fde_min_12 = 0, 0
    test_samples = 0

    batch_index = 0
    batch_size_accum = 0
    sub_batches = []
    frame_indexes = []

    for ii, batch_i in enumerate(test_batches):
        batch_size_accum += batch_i[1] - batch_i[0]
        test_samples += int(batch_i[1]- batch_i[0])
        sub_batches.append(batch_i)
        frame_indexes.append(train_size + ii)

        if ii >= test_size - 1 or \
                batch_size_accum + (test_batches[ii + 1][1] - test_batches[ii + 1][0]) > batch_size:
            
            bs = batch_size_accum
            # Slice the batch portion from all data
            batch_ids = dataset_agent_ids[sub_batches[0][0]:sub_batches[-1][1]]
            batch_obsv_5d = dataset_obsv_5d[sub_batches[0][0]:sub_batches[-1][1]]
            batch_pred_5d = dataset_pred_5d[sub_batches[0][0]:sub_batches[-1][1]]
            batch_fcn_features = all_fcn_features[frame_indexes[0]:frame_indexes[-1]+1]
            batch_social_features = test_social_features[batch_index]
            batch_horizon_angles = test_horizon_angles[batch_index]
            sub_batches = sub_batches - sub_batches[0][0]
            
            lstm_h_c = (torch.zeros(n_lstm_layers, bs, encoder.hidden_size).to(device1),
                        torch.zeros(n_lstm_layers, bs, encoder.hidden_size).to(device1))
            encoder.init_lstm(lstm_h_c[0], lstm_h_c[1])

            with torch.no_grad():
                encoder(batch_obsv_5d)
                batch_encoder_hidden_state = encoder.lstm_h[0]
                batch_encoder_cell_state = encoder.lstm_h[1]

                if use_visual:
                    batch_v_feature = v_self_attention(batch_fcn_features)
                    batch_v_feature = picture_sa_embedder(batch_v_feature)

                if use_social or use_horizon:
                    emb_features = feature_embedder(batch_social_features)
                    weighted_features = torch.zeros_like(encoder.lstm_h[0].squeeze())
                    hor_weighted_features = torch.zeros_like(encoder.lstm_h[0].squeeze())
                    if use_social:
                        weighted_features = attention(emb_features, encoder.lstm_h[0].squeeze(), sub_batches)
                    if use_horizon:
                        hor_weighted_features = hor_attention(emb_features, encoder.lstm_h[0].squeeze(), sub_batches, \
                                                            batch_social_features.data.cpu().numpy(), batch_horizon_angles.data.cpu().numpy())
                    batch_s_weighted_feature = weighted_features + hor_weighted_features
                else:
                    batch_s_weighted_feature = torch.zeros_like(encoder.lstm_h[0].squeeze())


                for i, sb in enumerate(sub_batches):
                    N = sb[1] - sb[0]
                    zeros = Variable(torch.zeros(N, 1) + np.random.uniform(0, 0.1), requires_grad=False).to(device1)
                    ones = Variable(torch.ones(N, 1) * np.random.uniform(0.9, 1.0), requires_grad=False).to(device1)

                    # Slice the frame portion from batch data
                    ids = batch_ids[sb[0]:sb[1]]
                    obsv_5d = batch_obsv_5d[sb[0]:sb[1]]
                    pred_5d = batch_pred_5d[sb[0]:sb[1]]
                    s_weighted_feature = batch_s_weighted_feature[sb[0]:sb[1]]
                    v_feature = batch_v_feature[i]

                    encoder.lstm_h = (batch_encoder_hidden_state[:, sb[0]:sb[1], :],\
                                    batch_encoder_cell_state[:, sb[0]:sb[1], :]) 

                    lstm_h_c = (torch.zeros(n_lstm_layers, N, decoder.hidden_size).to(device1),
                                torch.zeros(n_lstm_layers, N, decoder.hidden_size).to(device1))
                    decoder.init_lstm(lstm_h_c[0], lstm_h_c[1])

                    dec_h = dec_h_register.pop(ids, device1)
                    v_weighted_feature = v_attention(v_feature, dec_h.squeeze())

                    
                    all_20_errors = []
                    all_20_preds = []

                    for kk in range(n_gen_samples):
                        noise = torch.FloatTensor(torch.rand(N, noise_len)).to(device1)
                        pred_hat_5d = generator(noise, encoder, s_weighted_feature, v_weighted_feature, obsv_5d)
                        if kk == n_gen_samples - 1:
                            dec_h_register.save(ids, decoder.lstm_h[0])

                        all_20_preds.append(pred_hat_5d.unsqueeze(0))
                        err_all = torch.pow((pred_hat_5d[:, :, :2] - pred_5d[:,:,:2]) / ss, 2).sum(dim=2, keepdim=True).sqrt()
                        all_20_errors.append(err_all.unsqueeze(0))
                    
                    all_20_errors = torch.cat(all_20_errors)

                    # =============== Prediction Errors ================
                    fde_min_12_i, _ = all_20_errors[:, :, -1].min(0, keepdim=True)
                    ade_min_12_i, _ = all_20_errors.mean(2).min(0, keepdim=True)
                    fde_min_12 += fde_min_12_i.sum().item()
                    ade_min_12 += ade_min_12_i.sum().item()
                    fde_avg_12 += all_20_errors[:, :, -1].mean(0, keepdim=True).sum().item()
                    ade_avg_12 += all_20_errors.mean(2).mean(0, keepdim=True).sum().item()
                    # ==================================================


            batch_index = batch_index + 1
            batch_size_accum = 0
            sub_batches = []
            frame_indexes = []


    print("test samples:", test_samples)
    ade_avg_12 /= test_samples
    fde_avg_12 /= test_samples
    ade_min_12 /= test_samples
    fde_min_12 /= test_samples
    print('Avg ADE,FDE (12)= (%.3f, %.3f) | Min(20) ADE,FDE (12)= (%.3f, %.3f)' \
          % (ade_avg_12, fde_avg_12, ade_min_12, fde_min_12))

    test_file_name = os.path.join(evaluate_result_dir, "test.txt")
    f_val = open(test_file_name, "a")
    if epoch:
        lines = 'Epoch(%d): Avg ADE,FDE (12)= (%.3f, %.3f) | Min(20) ADE,FDE (12)= (%.3f, %.3f)\n' \
            % (epoch, ade_avg_12, fde_avg_12, ade_min_12, fde_min_12)
    else:
        lines = 'Test: Avg ADE,FDE (12)= (%.3f, %.3f) | Min(20) ADE,FDE (12)= (%.3f, %.3f)\n' \
            % (ade_avg_12, fde_avg_12, ade_min_12, fde_min_12)
    f_val.writelines(lines)
    f_val.close()
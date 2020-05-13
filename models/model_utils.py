import os
import torch


# Evaluate the error between the model prediction and the true path
def calc_error(pred_hat, pred):
    N = pred.size(0)
    T = pred.size(1)
    err_all = torch.pow((pred_hat - pred) / ss, 2).sum(dim=2).sqrt()  # N x T
    FDEs = err_all.sum(dim=0).item() / N
    ADEs = torch.cumsum(FDEs)
    for ii in range(T):
        ADEs[ii] /= (ii + 1)
    return ADEs.data.cpu().numpy(), FDEs.data().cpu().numpy()



class FutureHiddenStateRegister():
    def __init__(self, hidden_size, n_lstm_layers):
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.decoder_hidden_states = dict()
        self.preserved_decoder_hidden_states = dict()
    
    def reset(self):
        self.decoder_hidden_states.clear()

    def save(self, ids, dec_h):
        no_grad_dec_h = dec_h.clone().detach().cpu()
        list_dec_h = list(torch.split(no_grad_dec_h, 1, dim=1))
        dict_data = dict(list(zip(ids, list_dec_h)))
        self.decoder_hidden_states.update(dict_data)

    def pop(self, ids, device):
        dec_h = torch.zeros(self.n_lstm_layers, len(ids), self.hidden_size).to(device)
        for i, id in enumerate(ids):
            if id in list(self.decoder_hidden_states.keys()):
                dec_h[:, i, :] = self.decoder_hidden_states[id]
        return dec_h

    def preserve(self):
        self.preserved_decoder_hidden_states = self.decoder_hidden_states.copy()

    def reload(self):
        self.decoder_hidden_states = self.preserved_decoder_hidden_states



class ModelSelector():
    def __init__(self):
        self.evl_result_list = []


    def sort_evl_result(self, sorted_index=3):
        self.evl_result_list.sort(key=lambda elem: float(elem[sorted_index]))


    def save_evl_result(self, lines):
        for line in lines:
            line = line.split(' ')
            epoch = line[0][6:-2]
            avg_ade = line[4][1:-1]
            avg_fde = line[5][0:-1]
            min_ade = line[10][1:-1]
            min_fde = line[11][0:-2]
            self.evl_result_list.append((epoch, avg_ade, avg_fde, min_ade, min_fde))


    def get_best_model(self, model_name, dataset_name, evaluate_result_dir, model_checkpoints_dir):
        val_file_name = os.path.join(evaluate_result_dir, "val.txt")
        f_val = open(val_file_name, "r")
        self.save_evl_result(f_val.readlines())

        self.sort_evl_result()
        epoch = int(self.evl_result_list[0][0])
        avg_ade = float(self.evl_result_list[0][1])
        avg_fde = float(self.evl_result_list[0][2])
        min_ade = float(self.evl_result_list[0][3])
        min_fde = float(self.evl_result_list[0][4])

        model_path = model_checkpoints_dir + '/' + model_name + '_' + dataset_name + \
                     '_' + str(epoch) + '.pt'

        print('Best validation model is %d epoch. \n Avg ADE,FDE (12)= (%.3f, %.3f) | Min(20) ADE,FDE (12)= (%.3f, %.3f)' \
               % (epoch, avg_ade, avg_fde, min_ade, min_fde))

        return model_path

    
            


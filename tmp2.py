import os
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from Utils import Vocabulary, rm_voc_less, construct_voc
from Dataset import MolData, MolData2, MolData_pre
from Model import Generator, Discriminator
import tensorboardX
from tensorboardX import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse


class Predictor:
    def __init__(self, emb_size=128, convs=[(100, 1), (200, 2), (200, 3),
                                            (200, 4), (200, 5), (100, 6),
                                            (100, 7), (100, 8), (100, 9),
                                            (100, 10), (160, 15), (160, 20)], dropout=0.5,
                 n_epochs=100, lr=0.001,
                 load_dir=None, save_dir=None, log_dir=None,
                 log_every=100, save_every=500, voc=None, device=None):
        # def __init__(self, emb_size=128, n_layers=6, n_head=8, d_k=64, d_v=64, d_model=512, d_inner=2048, dropout=0.5,
        #              n_epochs=100, lr=0.001,
        #              load_dir=None, save_dir=None, log_dir=None,
        #              log_every=100, save_every=500, voc=None, device=None):
        self.voc = voc
        self.discriminator = Discriminator(voc, emb_size, convs, dropout=dropout)
        # self.discriminator = Discriminator(voc, emb_size, n_layers, n_head, d_k, d_v,
        #     d_model, d_inner, dropout=dropout)
        self.n_epochs = n_epochs
        self.lr = lr
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.log_every = log_every
        self.save_every = save_every
        if self.log_dir:
            self.writer = SummaryWriter(self.log_dir, flush_secs=10)
        if device:
            self.device = torch.device(device)
            self.discriminator = self.discriminator.to(self.device)
        else:
            self.device = device
        # Can restore from a saved RNN
        if load_dir:
            checkpoint = torch.load(load_dir)
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            else:
                raise Exception('%s already exist' % self.save_dir)

    def fit(self, train_set, valid_set):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        global_step = 0
        for epoch in range(1, self.n_epochs + 1):
            for train_step, batch in enumerate(train_set):
                global_step += 1
                inputs_from_data, labels = batch
                if self.device:
                    inputs_from_data = inputs_from_data.to(self.device)
                    labels = labels.to(self.device)
                outputs = self.discriminator(inputs_from_data)
                loss = criterion(outputs, labels.contiguous().view(-1, 1))
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.discriminator.parameters(), 5.)
                optimizer.step()
                if global_step % self.log_every == 0 or global_step == 1:
                    self.discriminator.eval()
                    train_out_metrics = self.evaluate(train_set)
                    valid_out_metrics = self.evaluate(valid_set)
                    for metric in train_out_metrics:
                        self.writer.add_scalars('%s' % metric, {'train': train_out_metrics[metric]}, global_step)
                        self.writer.add_scalars('%s' % metric, {'valid': valid_out_metrics[metric]}, global_step)
                    self.discriminator.train()
                if global_step % self.save_every == 0 or global_step == 1:
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'discriminator_state_dict': self.discriminator.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(self.save_dir, 'D_epoch_%s_step_%s.ckpt' % (epoch, global_step)))
        self.writer.close()

    def evaluate(self, valid_set, some_metrics=None, rl=False):
        criterion = nn.BCEWithLogitsLoss()
        with torch.no_grad():
            out_metrics = {}
            for eval_step, batch in enumerate(valid_set):
                inputs_from_data, labels = batch
                if self.device:
                    inputs_from_data = inputs_from_data.to(self.device)
                    labels = labels.to(self.device)
                logits = self.discriminator(inputs_from_data)
                tmp_logits = torch.sigmoid(logits).contiguous().view(-1).data.cpu().numpy()
                # tmp_logits = logits.data.cpu().numpy()
                loss = criterion(logits, labels.contiguous().view(-1, 1))
                pred = [1 if i >= 0.5 else 0 for i in tmp_logits]
                pre = precision_score(labels.data.cpu().numpy(), pred)
                rec = recall_score(labels.data.cpu().numpy(), pred)
                f1_ = f1_score(labels.data.cpu().numpy(), pred)
                out_metrics['loss'] = loss
                out_metrics['precision'] = pre
                out_metrics['recall'] = rec
                out_metrics['f1'] = f1_
                return out_metrics

    def predict(self, valid_set):
        self.discriminator.eval()
        with torch.no_grad():
            smi_ls = []
            logits_ls = []
            for eval_step, batch in enumerate(valid_set):
                # inputs_from_data, labels = batch
                inputs_from_data = batch
                seq_vec_ls = inputs_from_data.tolist()
                tmp_smi_ls = []
                for seq_vec in seq_vec_ls:
                    smile = self.voc.decode(seq_vec[1:])
                    tmp_smi_ls.append(smile)
                smi_ls.extend(tmp_smi_ls)
                if self.device:
                    inputs_from_data = inputs_from_data.to(self.device)
                    # labels = labels.to(self.device)
                logits = self.discriminator(inputs_from_data)
                tmp_logits = torch.sigmoid(logits).contiguous().view(-1).data.cpu().numpy()
                logits_ls.extend(tmp_logits)
            return smi_ls, logits_ls


def get_parser():
    parser = argparse.ArgumentParser(
        "Training initial discriminator"
    )
    parser.add_argument(
        '--infile_path', type=str, default='./BenchmarkDatasets/GA-sample10000.smi', help='Path to the dataset'
    )
    parser.add_argument(
        '--voc_path', type=str, default='./Datasets/Voc', help='Path to the Vocabulary'
    )
    parser.add_argument(
        '--visible_gpu', type=str, default='0', help='Visible GPU ids'
    )
    parser.add_argument(
        '--load_dir', type=str, default=None, help='Path to load model'
    )
    parser.add_argument(
        '--random_seed', type=float, default=666, help='Random seed for pytorch'
    )
    return parser


#########################
# predict
###################
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
torch.manual_seed(666)
voc_path = './Datasets/Voc'
voc = Vocabulary(init_from_file=voc_path, max_length=140)
esti = Predictor(emb_size=128, convs=[(100, 1), (200, 2), (200, 3),
                                      (200, 4), (200, 5), (100, 6),
                                      (100, 7), (100, 8), (100, 9),
                                      (100, 10), (160, 15), (160, 20)], dropout=0.5,
                 n_epochs=100, lr=0.0001,
                 load_dir='./AD_save/ADtrained_D.ckpt', save_dir=None, log_dir=None,
                 log_every=50, save_every=500, voc=voc, device='cuda:0')


def pre(infile):
    # out_file_name = infile + '_out.csv'
    # print(out_file_name)
    # all_df = pd.read_csv(infile, header=None)
    # x_train = all_df[0].values
    x_train=['COc1ccc2ccccc2c1CN1CCN(Cc2ccccc2)CC1']
    trian_data = MolData_pre(x_train, voc)
    train_set = DataLoader(trian_data, batch_size=1, shuffle=False, drop_last=False,
                           collate_fn=trian_data.collate_d)
    smi_ls, logits_ls = esti.predict(train_set)
    out_df = pd.DataFrame(smi_ls)
    out_df[1] = logits_ls
    out_df.columns = ['smiles', 'logits']
    print(out_df)
    # out_df.to_csv(out_file_name, index=False,header=None)
pre('aaa')
# names=['ALDH1', 'ESR-ANTAGTO', 'FEN1', 'GBA', 'KAT2A', 'MAPK1',
#        'PKM2', 'VDR']
# file_ls = ['PCBA/%s_active_T_rd_rm_less.smi'%name for name in names]+\
# ['PCBA/%s_active_V_rd_rm_less.smi'%name for name in names]+\
# ['PCBA/%s_inactive_T_rd_rm_less.smi'%name for name in names]+\
# ['PCBA/%s_inactive_V_rd_rm_less.smi'%name for name in names]
# for file in file_ls:
#     pre(file)



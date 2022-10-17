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
from Dataset import MolData,MolData2
from Model import Generator, Discriminator
import tensorboardX
from tensorboardX import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
#from metrics import Metrics
import argparse

class PretrainG:
    def __init__(self, emb_size=128, hidden_size=512, num_layers=3, dropout=0.5,
                 n_epochs=100, lr=0.001,
                 load_dir=None, save_dir=None, log_dir=None,
                 log_every=100, save_every=500, voc=None, device=None):
        self.voc = voc
        self.generator = Generator(self.voc, emb_size=emb_size, hidden_size=hidden_size,
                                   num_layers=num_layers, dropout=dropout)
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
            self.generator = self.generator.to(self.device)
        else:
            self.device = device
        # Can restore from a saved RNN
        if load_dir:
            checkpoint = torch.load(load_dir)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            else:
                raise Exception('%s already exist'%self.save_dir)

    def fit(self, train_set, valid_set):
        criterion = nn.CrossEntropyLoss(ignore_index=self.voc.vocab['PAD'])
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        global_step = 0
        for epoch in range(1, self.n_epochs+1):
            for train_step, batch in enumerate(train_set):
                global_step += 1
                tensors, prevs, nexts, lens = batch
                if self.device:
                    prevs = prevs.to(self.device)
                    nexts = nexts.to(self.device)
                    lens = lens.to(self.device)
                outputs, _, _ =self.generator(prevs, lens)
                loss = criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.generator.parameters(), 5.)
                optimizer.step()
                if global_step % self.log_every ==0 or global_step == 1:
                    self.generator.eval()
                    train_out_metrics = self.evaluate(train_set)
                    valid_out_metrics = self.evaluate(valid_set)
                    for metric in train_out_metrics:
                        self.writer.add_scalars('%s'%metric, {'train': train_out_metrics[metric]}, global_step)
                        self.writer.add_scalars('%s'%metric, {'valid': valid_out_metrics[metric]}, global_step)
                    out_smiles_ls = self.sample(n_batch=128)
                    print(out_smiles_ls)
                    valid = 0
                    for smi_i in out_smiles_ls:
                        try:
                            mol = Chem.MolFromSmiles(smi_i)
                            if mol:
                                valid += 1
                        except:
                            continue
                    self.writer.add_scalar('valid_smiles_rate', round(100 * valid / 128, 2), global_step)
                    self.generator.train()
                if global_step % self.save_every == 0 or global_step == 1:
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'generator_state_dict': self.generator.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(self.save_dir, 'G_epoch_%s_step_%s.ckpt'%(epoch, global_step)))
        self.writer.close()

    def evaluate(self, valid_set, some_metrics=None, rl=False):
        criterion = nn.CrossEntropyLoss(ignore_index=self.voc.vocab['PAD'])
        with torch.no_grad():
            out_metrics ={}
            for eval_step, batch in enumerate(valid_set):
                tensors, prevs, nexts, lens = batch
                # print(tensors[0])
                # print(self.voc.decode(tensors[0].numpy()))
                if self.device:
                    prevs = prevs.to(self.device)
                    nexts = nexts.to(self.device)
                    lens = lens.to(self.device)
                outputs, _, _ = self.generator(prevs, lens)
                # print(outputs[0].cpu().numpy())
                # print(self.voc.decode([np.argmax(i) for i in outputs[0].cpu().numpy()]))
                loss = criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))
                out_metrics['loss'] = loss
                return out_metrics
            
    def sample(self,n_batch,max_len=100):
        with torch.no_grad():
            prevs = torch.empty(n_batch, 1, dtype=torch.long, device=self.device).fill_(self.voc.vocab['GO'])
            n_sequences = prevs.shape[0]
            sequences = []
            lengths = torch.zeros(n_sequences, dtype=torch.long, device=prevs.device)
            one_lens = torch.ones(n_sequences, dtype=torch.long, device=prevs.device)
            is_end = prevs.eq(self.voc.vocab['EOS']).view(-1)
            states = None
            for _ in range(max_len):
                outputs, _, states = self.generator(prevs, one_lens, states)
                probs = F.softmax(outputs, dim=-1).view(n_sequences, -1)
                currents = torch.multinomial(probs, 1)
                currents[is_end, :] = self.voc.vocab['PAD']
                sequences.append(currents)
                lengths[~is_end] += 1
                is_end[currents.view(-1) == self.voc.vocab['EOS']] = 1
                if is_end.sum() == n_sequences:
                    break
                prevs = currents
            sequences = torch.cat(sequences, dim=-1)
            out_smiles = []
            seq_vec_ls = sequences.tolist()
            for seq_vec in seq_vec_ls:
                smile = self.voc.decode(seq_vec)
                out_smiles.append(smile)
            return out_smiles




def get_parser():
    parser = argparse.ArgumentParser(
        "Training initial generator"
    )
    parser.add_argument(
        '--infile_path', type=str, default='./Datasets/Data4InitG.smi', help='Path to the dataset'
    )
    parser.add_argument(
        '--voc_path', type=str, default='./Datasets/Voc', help='Path to the Vocabulary'
    )
    parser.add_argument(
        '--visible_gpu', type=str, default='0', help='Visible GPU ids'
    )
    parser.add_argument(
        '--model_save_path', type=str, default='pretrainG_save', help='Path for saving models'
    )
    parser.add_argument(
        '--save_every', type=float, default=1000, help='Model is saved after save_every steps'
    )
    parser.add_argument(
        '--log_path', type=str, default='pretrainG_log', help='Path for logging files'
    )
    parser.add_argument(
        '--log_every', type=float, default=100, help='Logging after log_every steps'
    )
    parser.add_argument(
        '--batch_size', type=float, default=512, help='Batch size'
    )
    parser.add_argument(
        '--n_epochs', type=float, default=100, help='Epochs'
    )
    parser.add_argument(
        '--lr', type=float, default=0.0001, help='Learning rate'
    )
    parser.add_argument(
        '--load_dir', type=str, default=None, help='Path to load model'
    )
    parser.add_argument(
        '--random_seed', type=float, default=666, help='Random seed for pytorch'
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    config, unknown = parser.parse_known_args()
    print(config)
    os.environ["CUDA_VISIBLE_DEVICES"]=config.visible_gpu
    # torch.manual_seed(config.random_seed)
    voc_path = config.voc_path
    voc = Vocabulary(init_from_file=voc_path)
    all_df = pd.read_csv(config.infile_path, header=None)
    x_valid = all_df.sample(100000, random_state=0)
    x_train = all_df.drop(x_valid.index)[0].values
    x_valid = x_valid[0].values
    trian_data = MolData(x_train, voc)
    train_set = DataLoader(trian_data, batch_size=config.batch_size, shuffle=True, drop_last=False, collate_fn=trian_data.collate_g)
    valid_data = MolData(x_valid, voc)
    valid_set = DataLoader(valid_data, batch_size=config.batch_size, shuffle=True, drop_last=False, collate_fn=valid_data.collate_g)

    esti = PretrainG(emb_size=128, hidden_size=512, num_layers=3, dropout=0.5,
                     n_epochs=config.n_epochs, lr=config.lr,
                     load_dir=config.load_dir, save_dir=config.model_save_path, log_dir=config.log_path,
                     log_every=config.log_every, save_every=config.save_every, voc=voc, device='cuda:0')
    esti.fit(train_set, valid_set)





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
import random
import copy
import argparse


class PolicyGradientLoss(nn.Module):
    def forward(self, outputs, targets, rewards, lengths):
        # log_probs = F.log_softmax(outputs, dim=2)
        log_probs = F.softmax(outputs, dim=2)
        # log_probs = outputs
        items = torch.gather(log_probs, 2, targets.unsqueeze(2)) * rewards.unsqueeze(2)
        loss = -sum([t[:l].sum()/l.float() for t, l in zip(items, lengths)]) / float(len(items))
        return loss


class PG:
    def __init__(self, emb_size=128, hidden_size=512, num_layers=3, dropout=0.5, convs=None,
                 lr=0.001,
                 load_dir=None,load_dir_G=None,load_dir_D=None, save_dir=None, log_dir=None,
                 voc=None, device=None):
        self.voc = voc
        self.generator = Generator(self.voc, emb_size=emb_size, hidden_size=hidden_size,
                                   num_layers=num_layers, dropout=dropout)
        self.discriminator = Discriminator(voc, emb_size, convs, dropout=dropout)
        self.lr = lr
        self.save_dir = save_dir
        self.log_dir = log_dir
        if self.log_dir:
            self.writer = SummaryWriter(self.log_dir, flush_secs=10)
        if device:
            self.device = torch.device(device)
            self.generator = self.generator.to(self.device)
            self.discriminator = self.discriminator.to(self.device)
        else:
            self.device = device
        # Can restore from a saved RNN
        if load_dir:
            checkpoint = torch.load(load_dir)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        if load_dir_G:
            checkpoint = torch.load(load_dir_G)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
        if load_dir_D:
            checkpoint = torch.load(load_dir_D)
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            else:
                raise Exception('%s already exist'%self.save_dir)

    def fit(self, train_set, valid_set=None, epochs=1000,
            g_update=1, d_update=1,early_stop=1,
            n_batch=128, rollouts=16,ref_smiles=None, ref_mols=None, max_length=100):
        g_criterion = PolicyGradientLoss()
        g2_criterion = nn.CrossEntropyLoss(ignore_index=self.voc.vocab['PAD'])
        d_criterion = nn.BCEWithLogitsLoss()
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        global_g_t=0
        global_d_t = 0
        best_d_loss=100
        patient=0
        iter=0
        for epoch in range(1, int(epochs+1)):
            for batch_id in range(1, len(train_set)):
                iter +=1
                print('iter %s is begining!!!' % iter)
                for g_i in range(g_update):
                    global_g_t += 1
                    print('g_step %s is begining!!!' % global_g_t)
                    self.generator.eval()
                    self.discriminator.eval()
                    sampled_batches = self.sample_tensor(n_batch, max_length)[0]
                    out_smiles = []
                    seq_vec_ls = sampled_batches.tolist()
                    for seq_vec in seq_vec_ls:
                        smile = self.voc.decode(seq_vec[1:])
                        out_smiles.append(smile)
                    print(out_smiles[:10])
                    sequences, rewards, lengths = self.rollout(n_batch, rollouts, ref_smiles, ref_mols, max_length)
                    self.generator.train()
                    self.discriminator.train()
                    lengths, indices = torch.sort(lengths, descending=True)
                    sequences = sequences[indices, ...]
                    rewards = rewards[indices, ...]
                    mean_episode_reward = sum([t[:l].mean() for t, l in zip(rewards, lengths)]) / float(len(rewards))
                    self.writer.add_scalar('g_episode_reward', mean_episode_reward, global_g_t)

                    generator_outputs, lengths, _ = self.generator(sequences[:, :-1], lengths - 1)
                    generator_loss_1 = g_criterion(generator_outputs, sequences[:, 1:], rewards, lengths)
                    self.writer.add_scalar('g1_loss', generator_loss_1, global_g_t)
                    if valid_set is not None:
                        for inputs_from_else in valid_set:
                            tensors, prevs, nexts, lens = inputs_from_else
                            if self.device:
                                prevs = prevs.to(self.device)
                                nexts = nexts.to(self.device)
                                lens = lens.to(self.device)
                            generator_outputs_else, lengths_else, _ = self.generator(prevs, lens)
                            generator_loss_2 = g2_criterion(generator_outputs_else.view(-1, generator_outputs_else.shape[-1]), nexts.view(-1))
                            self.writer.add_scalar('g2_loss', generator_loss_2, global_g_t)
                            break
                    if valid_set is not None:
                        generator_loss = (generator_loss_1 + generator_loss_2) / 2.
                    else:
                        generator_loss = generator_loss_1


                    self.writer.add_scalar('g_total_loss', generator_loss, global_g_t)

                    g_optimizer.zero_grad()
                    generator_loss.backward()
                    nn.utils.clip_grad_value_(self.generator.parameters(), 5.)
                    g_optimizer.step()
                    torch.save({
                        'iter': iter,
                        'generator_state_dict': self.generator.state_dict(),
                    }, os.path.join(self.save_dir, 'G_%s.ckpt'%iter))
                for d_i in range(d_update):
                    global_d_t += 1
                    print('d_step %s is begining!!!' % global_d_t)
                    self.generator.eval()
                    sampled_batches = self.sample_tensor(n_batch, max_length)[0]
                    out_smiles = []
                    seq_vec_ls = sampled_batches.tolist()
                    for seq_vec in seq_vec_ls:
                        smile = self.voc.decode(seq_vec[1:])
                        out_smiles.append(smile)
                    print(out_smiles[:10])
                    discrim_fake_outputs = self.discriminator(sampled_batches)
                    discrim_fake_targets = torch.zeros(len(discrim_fake_outputs), 1, device=self.device)
                    fake_loss = d_criterion(discrim_fake_outputs, discrim_fake_targets)
                    self.writer.add_scalar('fake_loss', fake_loss, global_d_t)
                    for inputs_from_data in train_set:
                        inputs_from_data = inputs_from_data.to(self.device)
                        discrim_real_outputs = self.discriminator(inputs_from_data)
                        discrim_real_targets = torch.ones(len(discrim_real_outputs), 1, device=self.device)
                        real_loss = d_criterion(discrim_real_outputs, discrim_real_targets)
                        self.writer.add_scalar('real_loss', real_loss, global_d_t)
                        break
                    discrim_loss = (fake_loss + real_loss) / 2.
                    tmp_d_loss = discrim_loss.item()
                    self.writer.add_scalar('d_loss', discrim_loss, global_d_t)
                    d_optimizer.zero_grad()
                    discrim_loss.backward()
                    nn.utils.clip_grad_value_(self.discriminator.parameters(), 5.)
                    d_optimizer.step()
                    torch.save({
                        'iter': iter,
                        'discriminator_state_dict': self.discriminator.state_dict(),
                    }, os.path.join(self.save_dir, 'D_%s.ckpt' % iter))
            if tmp_d_loss<best_d_loss:
                best_d_loss=tmp_d_loss
                patient=0
            else:
                patient+=1
            if patient>=early_stop:
                break

        self.writer.close()

    def _proceed_sequences(self, prevs, states, max_len):
        with torch.no_grad():
            n_sequences = prevs.shape[0]
            sequences = []
            lengths = torch.zeros(n_sequences, dtype=torch.long, device=prevs.device)
            one_lens = torch.ones(n_sequences, dtype=torch.long, device=prevs.device)
            is_end = prevs.eq(self.voc.vocab['EOS']).view(-1)
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
        return sequences, lengths

    def rollout(self, n_samples, n_rollouts, ref_smiles=None, ref_mols=None, max_len=100):
        with torch.no_grad():
            sequences = []
            rewards = []
            lengths = torch.zeros(n_samples, dtype=torch.long, device=self.device)
            one_lens = torch.ones(n_samples, dtype=torch.long, device=self.device)
            prevs = torch.empty(n_samples, 1, dtype=torch.long, device=self.device).fill_(self.voc.vocab['GO'])
            is_end = torch.zeros(n_samples, dtype=torch.bool, device=self.device)
            states = None
            sequences.append(prevs)
            lengths += 1
            for current_len in range(max_len):
                outputs, _, states = self.generator(prevs, one_lens, states)
                probs = F.softmax(outputs, dim=-1).view(n_samples, -1)
                currents = torch.multinomial(probs, 1)
                currents[is_end, :] = self.voc.vocab['PAD']
                sequences.append(currents)
                lengths[~is_end] += 1
                rollout_prevs = currents[~is_end, :].repeat(n_rollouts, 1)
                rollout_states = (
                    states[0][:, ~is_end, :].repeat(1, n_rollouts, 1),
                    states[1][:, ~is_end, :].repeat(1, n_rollouts, 1)
                )
                rollout_sequences, rollout_lengths = self._proceed_sequences(
                    rollout_prevs, rollout_states, max_len - current_len
                )
                rollout_sequences = torch.cat( [s[~is_end, :].repeat(n_rollouts, 1)
                     for s in sequences] + [rollout_sequences], dim=-1
                )
                rollout_lengths += lengths[~is_end].repeat(n_rollouts)
                rollout_rewards = torch.sigmoid(self.discriminator(rollout_sequences).detach())
                current_rewards = torch.zeros(n_samples, device=self.device)
                current_rewards[~is_end] = rollout_rewards.view(n_rollouts, -1).mean(dim=0)
                rewards.append(current_rewards.view(-1, 1))
                is_end[currents.view(-1) == self.voc.vocab['EOS']] = 1
                if is_end.sum() == n_samples:
                    break
                prevs = currents
            sequences = torch.cat(sequences, dim=1)
            rewards = torch.cat(rewards, dim=1)
        return sequences, rewards, lengths

    def sample_tensor(self, n, max_len=100):
        prevs = torch.empty(n, 1, dtype=torch.long, device=self.device).fill_(self.voc.vocab['GO'])
        samples, lengths = self._proceed_sequences(prevs, None, max_len)
        samples = torch.cat([prevs, samples], dim=-1)
        lengths += 1
        return samples, lengths


def get_parser():
    parser = argparse.ArgumentParser(
        "Adversarial Training"
    )
    parser.add_argument(
        '--infile_path', type=str, default='./Datasets/Data4InitD.txt', help='Path to the dataset'
    )
    parser.add_argument(
        '--voc_path', type=str, default='./Datasets/Voc', help='Path to the Vocabulary'
    )
    parser.add_argument(
        '--visible_gpu', type=str, default='0', help='Visible GPU ids'
    )
    parser.add_argument(
        '--model_save_path', type=str, default='AD_save', help='Path for saving models'
    )
    parser.add_argument(
        '--log_path', type=str, default='AD_log', help='Path for logging files'
    )
    parser.add_argument(
        '--batch_size', type=float, default=64, help='Batch size'
    )
    parser.add_argument(
        '--epochs', type=float, default=10, help='Epochs'
    )
    parser.add_argument(
        '--lr', type=float, default=0.0001, help='Learning rate'
    )
    parser.add_argument(
        '--load_dir_G', type=str, default='pretrainG_save/pretrained_G.ckpt', help='Path to load init G'
    )
    parser.add_argument(
        '--load_dir_D', type=str, default='pretrainD_save/pretrained_D.ckpt', help='Path to load init D'
    )
    parser.add_argument(
        '--g_update', type=float, default=1, help='The generator is updated every g_update steps'
    )
    parser.add_argument(
        '--d_update', type=float, default=1, help='The discriminator is updated every g_update steps'
    )
    parser.add_argument(
        '--rollouts', type=float, default=16, help='N Monte Carlo searches'
    )
    parser.add_argument(
        '--early_stop', type=float, default=1, help='Early stopping epoch value'
    )
    parser.add_argument(
        '--drugset_tune', type=float, default=1, help='Whether fine tune generator with drugset'
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

    all_df = pd.read_csv(config.infile_path).values
    # print(all_df)
    np.random.seed(0)
    num_datapoints = len(all_df)
    train_cutoff = int(0.8 * num_datapoints)
    valid_cutoff = int((0.8 + 0.1) * num_datapoints)
    shuffled = np.random.permutation(range(num_datapoints))
    # print(shuffled)
    train_d = all_df[shuffled[:train_cutoff]]
    tmp_df = pd.DataFrame(train_d)
    # print(tmp_df)
    tmp_df = tmp_df[tmp_df[1]==1]
    print(tmp_df)
    x_train = tmp_df[0].values
    trian_data = MolData(x_train, voc)
    train_set = DataLoader(trian_data, batch_size=config.batch_size, shuffle=True, drop_last=True, collate_fn=trian_data.collate_d)

    valid_data = MolData(x_train, voc)
    valid_set = DataLoader(valid_data, batch_size=config.batch_size, shuffle=True, drop_last=True, collate_fn=valid_data.collate_g)
    #
    esti = PG(emb_size=128, hidden_size=512, num_layers=3, dropout=0.5,convs=[(100, 1), (200, 2), (200, 3),
                                        (200, 4), (200, 5), (100, 6),
                                        (100, 7), (100, 8), (100, 9),
                                        (100, 10), (160, 15), (160, 20)],
                     lr=config.lr,
                     load_dir=None, load_dir_G=config.load_dir_G,
              load_dir_D=config.load_dir_D,save_dir=config.model_save_path, log_dir=config.log_path,
                     voc=voc, device='cuda:0')
    if config.drugset_tune==1.0:
        esti.fit(train_set, valid_set=valid_set, epochs=config.epochs,
                    g_update=config.g_update, d_update=config.d_update,
                    n_batch=config.batch_size, rollouts=config.rollouts, early_stop=config.early_stop)
    else:
        esti.fit(train_set, valid_set=None, epochs=config.epochs,
                 g_update=config.g_update, d_update=config.d_update,
                 n_batch=config.batch_size, rollouts=config.rollouts, early_stop=config.early_stop)
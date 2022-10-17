import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np


class MolData(Dataset):
    """Custom PyTorch Dataset that takes a file containing SMILES.

        Args:
                fname : path to a file containing \n separated SMILES.
                voc   : a Vocabulary instance

        Returns:
                A custom PyTorch dataset for training the Prior.
    """
    def __init__(self, smiles_ls, voc):
        self.voc = voc
        self.smiles = pd.DataFrame(smiles_ls)[0].values

    def __getitem__(self, i):
        mol = self.smiles[i]
        tokenized = self.voc.tokenize(mol)
        encoded = self.voc.encode(tokenized)
        return encoded

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))

    def collate_g(cls, data):
        # print(data)
        data.sort(key=len, reverse=True)
        tensors = [torch.tensor(arr, dtype=torch.long) for arr in data]
        prevs = pad_sequence([t[:-1] for t in tensors], batch_first=True, padding_value=0)
        nexts = pad_sequence([t[1:] for t in tensors], batch_first=True, padding_value=0)
        lens = torch.tensor([len(t) - 1 for t in tensors], dtype=torch.long)
        return tensors, prevs, nexts, lens

    def collate_d(cls, data):
        data.sort(key=len, reverse=True)
        tensors = [torch.tensor(arr, dtype=torch.long) for arr in data]
        tensors_in = pad_sequence(tensors, batch_first=True, padding_value=0)
        return tensors_in

class MolData2(Dataset):
    """Custom PyTorch Dataset that takes a file containing SMILES.

        Args:
                fname : path to a file containing \n separated SMILES.
                voc   : a Vocabulary instance

        Returns:
                A custom PyTorch dataset for training the Prior.
    """
    def __init__(self, smiles_ls,label_ls, voc):
        self.voc = voc
        self.smiles = pd.DataFrame(smiles_ls)[0].values
        self.labels = pd.DataFrame(label_ls)[0].values

    def __getitem__(self, i):
        mol = self.smiles[i]
        tmp_label = self.labels[i]
        tokenized = self.voc.tokenize(mol)
        encoded = self.voc.encode(tokenized)
        return encoded, tmp_label

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))

    def collate(cls, data):
        # print(data)
        tmp_s=[]
        tmp_l=[]
        for i in data:
            tmp_s.append(i[0])
            tmp_l.append(i[1])
        tmp_s_arr = np.array(tmp_s)
        tmp_l_arr = np.array(tmp_l)
        # lens_ls = [len(i) for i in tmp_s]
        # sorted_ids=np.argsort(lens_ls)[::-1]
        # sorted_s = tmp_s_arr[sorted_ids]
        # sorted_l = tmp_l_arr[sorted_ids]

        # tensors = [torch.tensor(arr, dtype=torch.long) for arr in sorted_s]
        # tensors_in = pad_sequence(tensors, batch_first=True, padding_value=0)
        # lab = torch.tensor(sorted_l, dtype=torch.float)
        tensors = [torch.tensor(arr, dtype=torch.long) for arr in tmp_s_arr]
        tensors_in = pad_sequence(tensors, batch_first=True, padding_value=0)
        lab = torch.tensor(tmp_l_arr, dtype=torch.float)
        return tensors_in, lab
class MolData_pre(Dataset):
    """Custom PyTorch Dataset that takes a file containing SMILES.

        Args:
                fname : path to a file containing \n separated SMILES.
                voc   : a Vocabulary instance

        Returns:
                A custom PyTorch dataset for training the Prior.
    """
    def __init__(self, smiles_ls, voc):
        self.voc = voc
        self.smiles = pd.DataFrame(smiles_ls)[0].values

    def __getitem__(self, i):
        mol = self.smiles[i]
        tokenized = self.voc.tokenize(mol)
        # if sequence is too short an error will raise for CNN, so we pad the sequence to 20
        if len(tokenized)<20:
            tokenized += ['PAD']*(20-len(tokenized))
        encoded = self.voc.encode(tokenized)
        return encoded

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))

    def collate_d(cls, data):
        data.sort(key=len, reverse=True)
        tensors = [torch.tensor(arr, dtype=torch.long) for arr in data]
        tensors_in = pad_sequence(tensors, batch_first=True, padding_value=0)
        return tensors_in


# from torch.utils.data import DataLoader
# from rnn_utils import Vocabulary
# df=pd.DataFrame()
# df['smiles']=['CCCCCCCCCCCCCCCC','CCCCC','CCCCCCCC']
# df['label']=[1,0,1]
# voc_path = './Datasets/Voc'
# voc = Vocabulary(init_from_file=voc_path, max_length=140)
# d = MolData2(df['smiles'].values,df['label'].values,voc)
# train_set = DataLoader(d, batch_size=3, shuffle=False, drop_last=False, collate_fn=d.collate)
# for i in train_set:
#     print(i)
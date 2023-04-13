import rdkit
from rdkit import Chem
import re
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


file_path = os.path.abspath(os.path.dirname(__file__))
def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)
    return string
def construct_voc(smiles_list, min_str_freq=1):
    """Returns all the characters present in a SMILES file.
       Uses regex to find characters/tokens of the format '[x]'."""
    add_chars = {}
    for i, smiles in enumerate(smiles_list):
        regex = '(\[[^\[\]]{1,10}\])'
        smiles = replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                if char not in add_chars:
                    add_chars[char] = 1
                else:
                    add_chars[char] += 1
            else:
                chars = [unit for unit in char]
                for unit in chars:
                    if unit not in add_chars:
                        add_chars[unit] = 1
                    else:
                        add_chars[unit] += 1
    print("Number of characters: {}".format(len(add_chars)))
    res = sorted(add_chars.items(), key=lambda add_chars: add_chars[1], reverse=True)
    print(res)
    voc_ls = []
    less_ls = []
    for i in res:
        if i[1] > min_str_freq:
            voc_ls.append(i[0])
        else:
            less_ls.append(i[0])
    # with open(os.path.join(file_path,'Voc'), 'w') as f:
    #     for char in voc_ls:
    #         f.write(char + "\n")
    return voc_ls, less_ls

def rm_voc_less(smiles_list, voc_ls):
    smiles_list_final = []
    for smiles in smiles_list:
        regex = '(\[[^\[\]]{1,10}\])'
        smiles_ = replace_halogen(smiles)
        char_list = re.split(regex, smiles_)
        label = True
        for char in char_list:
            if char.startswith('['):
                if char not in voc_ls:
                    label = False
                    print(smiles)
                    break
            else:
                chars = [unit for unit in char]
                for unit in chars:
                    if unit not in voc_ls:
                        label = False
                        print(smiles)
                        break
        if label:
            smiles_list_final.append(smiles)
    return smiles_list_final

class Vocabulary(object):
    """A class for handling encoding/decoding from SMILES to an array of indices"""
    def __init__(self, init_from_file=None, max_length=140):
        self.special_tokens = ['PAD', 'GO', 'EOS']
        self.additional_chars = set()
        self.chars = self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length
        if init_from_file: self.init_from_file(init_from_file)

    def encode(self, char_list):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            if char in self.vocab:
                smiles_matrix[i] = self.vocab[char]
            else:
                smiles_matrix[i] = self.vocab['PAD']
        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if (i == self.vocab['EOS']) or (i == self.vocab['PAD']): break
            chars.append(self.reversed_vocab[i])
        smiles = "".join(chars)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def tokenize(self, smiles):
        """Takes a SMILES and return a list of characters/tokens"""
        regex = '(\[[^\[\]]{1,10}\])'
        smiles = replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        tokenized = []
        tokenized.append('GO')
        for char in char_list:
            if char.startswith('['):
                tokenized.append(char)
            else:
                chars = [unit for unit in char]
                [tokenized.append(unit) for unit in chars]
        tokenized.append('EOS')
        return tokenized

    def add_characters(self, chars):
        """Adds characters to the vocabulary"""
        for char in chars:
            self.additional_chars.add(char)
        char_list = list(self.additional_chars)
        char_list.sort()
        self.chars = self.special_tokens + char_list
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, 'r') as f:
            chars = f.read().split()
        self.add_characters(chars)

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return "Vocabulary containing {} tokens: {}".format(len(self), self.chars)

import rdkit
from rdkit import Chem
from rdkit.Chem.MolStandardize import charge
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
import joblib
from joblib import delayed
import re
import os
import argparse

def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)
    return string

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
                    #print(smiles)
                    break
            else:
                chars = [unit for unit in char]
                for unit in chars:
                    if unit not in voc_ls:
                        label = False
                        #print(smiles)
                        break
        smiles_list_final.append(label)
    return smiles_list_final


allowed_elements = ['H','C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
rules=['[CH2;!R][CH2;!R][CH2;!R][CH2;!R]', '[O].[O].[O].[O].[O].[O].[O].[O].[O].[O]']
non_s=rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED
rules_smarts = [Chem.MolFromSmarts(smarts) for smarts in rules]
pool = joblib.Parallel(n_jobs=20)

def check_smi(insmi):
    label = True
    mol = Chem.MolFromSmiles(insmi)
    if mol:
        atms = mol.GetAtoms()
        n_atoms = len(atms)
        if n_atoms < 10:
            label = False
        s_n = 0
        for atom in atms:
            if atom.GetSymbol() not in allowed_elements:
                label = False
                break
            if atom.GetIsotope():
                label = False
                break
            if atom.GetChiralTag() != non_s:
                s_n += 1
        if s_n >= 5:
            label = False

        for patt in rules_smarts:
            if len(mol.GetSubstructMatches(patt))>0:
                label = False
                break
        # check MW
        mw = Descriptors.MolWt(mol)
        if mw>750:
            label = False
        r_info = mol.GetRingInfo()
        if r_info.NumRings()>7:
            label = False
        r_a_size = [len(i) for i in r_info.AtomRings()]
        if r_a_size:
            max_r_size = max(r_a_size)
            min_r_size = min(r_a_size)
            if max_r_size>8:
                label=False
    else:
        label = False

    return label

def rd_mol(insmi):
    mol =Chem.MolFromSmiles(insmi)
    if mol:
        return Chem.MolToSmiles(mol,isomericSmiles=True, canonical=True)
    else:
        return None


def get_parser():
    parser = argparse.ArgumentParser(
        "Preprocessing steps"
    )
    parser.add_argument(
        '--infile_name', type=str, default='test.csv', help='Path to the input file'
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    config, unknown = parser.parse_known_args()
    # print(config)
    Infile_name=config.infile_name
    print(Infile_name)
    in_smi=[]
    df=pd.read_csv(Infile_name,header=None)
    mols_rd = df[0].values
    # with open(Infile_name) as f:
    #     for i in f:
    #         in_smi.append(i.rstrip())
    # print('Input file contains %s mols.'%len(in_smi))
    # rd molecules
    # mols_rd = pool(delayed(rd_mol)(s) for s in in_smi)
    # df[2]=mols_rd
    # mols_rd = [i for i in mols_rd if i is not None]
    # Remove molecules contains elements other than H, C, N, O, F, P, S, Cl, Br, I ,isotope and rules filtering
    #check_results=pool(delayed(check_smi)(s) for s in mols_rd)
    #df[2] = check_results
    # check_pass_smi=[mols_rd[i] for i,j in enumerate(check_results) if j]
    # Remove records containing more than two molecules
    #check_pass_smi=[i for i in mols_rd if '.' not in i]
    # remove duplicate molecules
    #check_pass_smi = list(set(check_pass_smi))
    #remove smiles contain str not in vocabulary
    voc_ls = []
    with open('./Datasets/Voc') as f:
        for i in f:
            voc_ls.append(i.rstrip())
    out_smi_ls = rm_voc_less(mols_rd, voc_ls)
    df[2] = out_smi_ls
    print(df)
    outdf=df[df[2]==True]
    #outdf=outdf[[0,1]]
    outdf.to_csv('results/DDR1_rm_voc_less.csv',index=False,header=None)
    print(outdf)
    # print('Output file contains %s mols.'%len(out_smi_ls))
    # with open('%s_preprocessed.csv'%Infile_name,'w') as f:
    #     for i in out_smi_ls:
    #         f.write(i+'\n')



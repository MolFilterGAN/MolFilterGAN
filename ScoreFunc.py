# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:01:43 2019

@author: xhliu
"""
try:
    import rdkit
    from rdkit import Chem
    from rdkit import rdBase
    from rdkit.Chem.rdchem import HybridizationType
    from rdkit.Chem import rdMolDescriptors as ds
    import numpy as np
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures,Lipinski
    from rdkit.Chem.rdchem import BondType as BT
    import sascorer
    from rdkit.Chem import ChemicalFeatures,Descriptors,rdMolDescriptors,QED
    rdBase.DisableLog('rdApp.error')
except ImportError:
    rdkit = None


def rule5(infile,outfile):
    a=open(infile)
    b=open(outfile,'w')
    for i in a:
        smiles=i.rstrip('\n')
        mol=Chem.MolFromSmiles(smiles)
        if mol:
            logp=Descriptors.MolLogP(mol)
            mw=Descriptors.MolWt(mol)
            hbd=rdMolDescriptors.CalcNumLipinskiHBD(mol)
            hda=rdMolDescriptors.CalcNumLipinskiHBA(mol)
            rb=rdMolDescriptors.CalcNumRotatableBonds(mol)
            b.write(smiles+','+str(logp)+','+str(mw)+','+str(hbd)+','+str(hda)+','+str(rb)+'\n')
    a.close()
    b.close()


def ca_sa(insmi):
    mol = Chem.MolFromSmiles(insmi)
    s = sascorer.calculateScore(mol)
    return s


def ca_qed(insmi):
    mol = Chem.MolFromSmiles(insmi)
    q = QED.qed(mol)
    return q


def CalculateZagreb1(mol):
    """
    #################################################################
    Calculation of Zagreb index with order 1 in a molecule

    ---->ZM1

    Usage:

        result=CalculateZagreb1(mol)

        Input: mol is a molecule object

        Output: result is a numeric value
    #################################################################
    """

    deltas = [x.GetDegree() for x in mol.GetAtoms()]
    return sum(np.array(deltas) ** 2)


def CalculateQuadratic(mol):
    """
    #################################################################
    Calculation of Quadratic index in a molecule

    ---->Qindex

    Usage:

        result=CalculateQuadratic(mol)

        Input: mol is a molecule object

        Output: result is a numeric value
    #################################################################
    """
    deltas = [x.GetDegree() for x in mol.GetAtoms()]
    M = sum(np.array(deltas) ** 2)
    N = mol.GetNumAtoms()
    return 3 - 2 * N + M / 2.0


def ca_MCE18(in_smi):
    mce_score = 0
    mol = Chem.MolFromSmiles(in_smi)
    AR = ds.CalcNumAromaticRings(mol)
    # print('AR:%s'%AR)
    if AR > 0:
        mce_score += 1
    NAR = ds.CalcNumAliphaticRings(mol)
    # print('NAR:%s'%NAR)
    if NAR > 0:
        mce_score += 1
    chiral = Chem.FindMolChiralCenters(mol)
    # print('chiral:%s' % chiral)
    if len(chiral) > 0:
        mce_score += 1
    spiro = ds.CalcNumSpiroAtoms(mol)
    # print('spiro:%s' % spiro)
    if spiro > 0:
        mce_score += 1

    sp3 = ds.CalcFractionCSP3(mol)
    # print('csp3:%s' % sp3)
    atom_num = 0
    C_num = 0
    csp3_num = 0
    csp3_cyc_num = 0
    csp3_acyc_num = 0
    for atom in mol.GetAtoms():
        atom_num += 1
        if atom.GetSymbol() == 'C':
            C_num += 1
            if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3:
                csp3_num += 1
                if atom.IsInRing():
                    csp3_cyc_num += 1
                else:
                    csp3_acyc_num += 1
            else:
                pass
        else:
            pass
    # csp3_C_num = csp3_cyc_num + csp3_acyc_num
    # cyc = csp3_cyc_num/csp3_num
    # acyc = csp3_acyc_num/csp3_num
    cyc = csp3_cyc_num / C_num
    acyc = csp3_acyc_num/C_num
    NCSPTR = (sp3 + cyc - acyc)/(1+sp3)
    mce_score += NCSPTR
    # print(mce_score)
    normed_q = CalculateQuadratic(mol)
    mce_score = mce_score * normed_q
    return mce_score


def ca_fsp3(insmi):
    mol = Chem.MolFromSmiles(insmi)
    p3 = Lipinski.FractionCSP3(mol)
    return p3




from ScoreFunc import ca_qed, ca_sa, ca_fsp3, ca_MCE18
import joblib
from joblib import delayed
import pandas as pd

pool = joblib.Parallel(n_jobs=40)


name_ls = ['GA-sample10000.smi', 'vae-zinc-s-sample10000.smi', 'lstm-zinc-sample10000.smi',
           'lstm-chembl-sample10000.smi', 'zinc-sample10000.smi', 'real-sample10000.smi',
           'chembl-sample10000.smi', 'cnpd-sample10000.smi', 'cortellis-drugs.smi']
name_ls=['./BenchmarkDatasets/' + i for i in name_ls]
print(name_ls)
#qed
cols_ls=['GA','vae-zinc-s','lstm-zinc','lstm-chembl','zinc','real','chembl','cnpd','cortellis-drugs']
df=pd.DataFrame()
for i,j in zip(name_ls,cols_ls):
    print('%s finished!'%i)
    smiles_ls = pd.read_csv(i,header=None)[0].values
    scores = pool(delayed(ca_MCE18)(s) for s in smiles_ls)
    if j=='cortellis-drugs':
        df[j]=[round(t,3) for t in scores]+[None]*(10000-748)
    else:
        df[j]=[round(t,3) for t in scores]
df.to_csv('results/MCE18.csv',index=False,na_rep='-')
# pool(delayed(rm_s)(s) for s in out_smi)

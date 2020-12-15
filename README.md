### MolFilterGAN
A Generative Adversarial Network-based Molecule Filtering Model for AI-designed Molecules 

## Requirement
This project requires the following libraries.
- NumPy
- Pandas
- PyTorch > 1.2
- RDKit

## Benchmark Datasets
`BenchmarkDatasets/` contains all the benchmark dataset for evaluating metrics.

## Datasets
`Datasets/` contains all the datasets for training MolFilterGAN

## pretrainG_save
`pretrainG_save/` contains a trained initial generator
## pretrainD_save
`pretrainD_save/` contains a trained initial discriminator

## AD_save
`AD_save/` contains a adversarial trained discriminator

## Training a initial generator
python PretrainG.py --infile_path Datasets/Data4InitG.smi --log_path test_init_G_log --model_save_path test_init_G_save

## Training a initial discriminator
python PretrainD.py --infile_path Datasets/Data4InitD.txt --log_path test_init_D_log --model_save_path test_init_D_save

## Adversarial Training
python AdversarialTraining.py --infile_path Datasets/Data4InitD.txt --log_path test_AD_log --model_save_path test_AD_save --load_dir_G pretrainG_save/pretrained_G.ckpt --load_dir_D pretrainD_save/pretrained_D.ckpt







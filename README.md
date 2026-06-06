# Rabies Risk ML Peru

Reproducible ML pipeline for zoonotic wild rabies transmission 
risk prediction in Peru using a One Health approach.

**Author:** Jorge Luis Limo Arispe - UNMSM Doctoral Program 2026

## What this is
Logistic regression baseline trained on simulated One Health 
variables (NDVI, temperature, precipitation, forest loss, 
bat occurrence) as a reproducible foundation for the doctoral 
research protocol.

## Data
`data/rabies_data.csv` is tracked with DVC.
Pointer file: `data/rabies_data.csv.dvc`

## Reproduce the result
1. pip install -r requirements.txt
2. dvc pull
3. python src/train.py --seed 42

## Expected output
seed=42  AUC-ROC=0.XXXX  accuracy=0.XXXX

## Run all experiments
python src/run_experiments.py

## Environment
Python 3.11 · exact packages in requirements.txt · see Dockerfile
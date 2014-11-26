Code for TFML 2014 conference

# Fit new SVM

To fit svms run `python fit_svms.py`. 

```
Usage: fit_svms.py [options]

Options:
  -h, --help            show this help message and exit
  -e EXPERIMENT_NAME, --e_name=EXPERIMENT_NAME
  --kernel=KERNEL       
  --experiment_name=EXPERIMENT_NAME
  --seed=SEED           
  --use_embedding=USE_EMBEDDING
  --fingerprint=FINGERPRINT
  --n_folds=N_FOLDS     
  --protein=PROTEIN     
  --max_hashes=MAX_HASHES
  --grid_w=GRID_W       
  --K=K        
```

Example

```
python fit_svms.py --kernel=linear --use_embedding=1 --protein=0 --fingerprint=4 --e_name=my_favourite_experiment
```

# Print and plot results

```
python scripts/fit_svms_print_results.py my_favourite_experiment
```


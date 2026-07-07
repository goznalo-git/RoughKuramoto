# Structure of this directory (Simulations_3D)


## Integration scripts and helper functions
- int_col.py
- int_sto.py

## Launchers and utilities
- sim_launcher.sh: tweak the parameters (L, M, K) inside, and then run as is. It calls with nohup both int_col.py and int_sto.py with the appropriate parameters.
- output_merger.py: run like `python output_merger.py OutputColumnar OutputTimeDep` to generate files at the corresponding folder, for each delta, e.g. OutputColumnar/all_simulations_delta0.pkl. **Important note:** running this script resets those files, erasing any observable computed from them in the notebooks of the parent directory.

## Outputs
- average_simulations.pkl: a product of the analysis notebooks. It contains the averaged observables (Roughness, Correlations, Structure factor, Covariance).
- Within this directory there is the following substructure of directories and files with the simulation data
  - OutputColumnar/
    - all_simulations_delta0.pkl
    - all_simulations_deltaatan5.pkl
    - delta0/
      - L{L}_K{K}/* 
    - deltaatan5/
      - L{L}_K{K}/*
  - OutputColumnar/
    - all_simulations_delta0.pkl
    - all_simulations_deltaatan5.pkl
    - delta0/
      - L{L}_K{K}/* 
    - deltaatan5/
      - L{L}_K{K}/*
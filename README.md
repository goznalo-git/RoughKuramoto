# RoughKuramoto

Python simulations of the Kuramoto-Sakaguchi model viewed as an interface growth phenomena.

## File list

Basic code files
- README.md
- .gitignore

`Visualizations/` directory
- animation_script.py
- deterministic_terms.ipynb
- gaussian_universality.ipynb

Notebooks for analysis of the simulation data at `Simulations/`. The final figures are saved at `Figures/`
- 1-Kuramoto-Sakaguchi.ipynb 
- 2-Roughness.ipynb
- 3-Correlations.ipynb
- 4-StructureFactor.ipynb
- 5-Covariance.ipynb


Helpers and custom functions for the notebooks
- surface_functions.py
- plotting_helpers.py

`Simulations/` directory: the models are integrated here.
- int_col.py
- int_sto.py
- integration_functions.py 
- aux_functions.py
- sim_launcher.sh: tweak the parameters (L, M, K) inside, and then run as is. It calls with nohup both int_col.py and int_sto.py with the appropriate parameters.
- output_merger.py: run like `python output_merger.py OutputColumnar OutputTimeDep` to generate files at the corresponding folder, for each delta, e.g. OutputColumnar/all_simulations_delta0.pkl
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



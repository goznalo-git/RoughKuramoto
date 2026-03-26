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

`Notebooks_1D/` and `Notebooks_2D/`: the notebooks for analysis of the simulation data at `Simulations_1D/` and `Simulations_2D/`, respectively, are found here. See the `README.md` files within for more information.

`Simulations_1D/` and `Simulations_2D` directories: the models are integrated there. See the `README.md` files within for more information. 

Utilities shared between the simulations in 1D and 2D are found in and imported from
- integration_functions.py 
- aux_functions.py


## Simulation instructions

For the simulation stage,
1. Prerequisites (packages): numpy, numba.
2. Go to the appropriate simulation folder, `Simulations_*D`.
3. Run `./sim_launcher.sh` with the desired set of parameters to compute batches of solutions. 
4. Run `python output_merger.py OutputColumnar OutputTimeDep` to merge all simulations of each time into respective files.

For the analysis stage, 
1. Prerequisites (packages): numpy, pandas, matplotlib, scienceplots 
2. Go to the appropriate analysis folder, `Notebooks_*D`.
3. Load the surface observable functions from the respective files.
4. Load the plotting functions (evolution, collapse) from the respective files.

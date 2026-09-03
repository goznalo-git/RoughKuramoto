#! /bin/bash

###########################################################################
## Command explanation

# python -u: for returning output to log file in real time (unbuffered)
# int_sto.py/int_col.py: script name
# 1st argument: deltaname (0 or atan5)
# 2nd argument: L (system size, integer)
# 3rd argument: K (coupling constant, integer)
# 4th argument: M (number of seeds sampled, integer)
###########################################################################

# Modify this according to the system
source ../../../../scientific_kernel/bin/activate   

L_VALUES=(125 250)
tandeltaval=5 # (for the horizontal scan)
Kval=10 # (for the vertical scan)      
M=5


####################
## Large coupling ##
####################

echo "## Running simulations. Date: `date`"
echo "Parameters: $tandeltaval $Kval $M" 

for L in "${L_VALUES[@]}"; do
    echo "Launching simulation for L=$L at $(date)"
    
    # Run in background with nohup
    # We use a unique log name for each L
        
    # HORIZONTAL (fixed delta, increasing K)
    nohup python -u int_sto_horiz.py $tandeltaval $L XXXX $M > Logs/HORIZ/out_${L}_${tandeltaval}_${M}.log 2>&1 &
    sleep 2

    
    # VERTICAL (fixed K, increasing delta)
    # nohup python -u int_sto_vert.py XXXX $L $Kval $M  > Logs/VERT/out_${L}_$Kval_${M}.log 2>&1 &
    # sleep 2

    
done


echo "All simulations launched. Use 'ps aux | grep python' to monitor."

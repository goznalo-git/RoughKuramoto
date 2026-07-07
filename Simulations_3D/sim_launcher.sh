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
source ../../../scientific_kernel/bin/activate   

L_VALUES=(125)
Ksto=15
Kcol=60
M=10


####################
## Large coupling ##
####################

echo "## Running simulations. Date: `date`"
echo "Parameters: $Ksto $Kcol $M" 

for L in "${L_VALUES[@]}"; do
    echo "Launching simulation for L=$L at $(date)"
    
    # Run in background with nohup
    # We use a unique log name for each L
        
    # Time dependent
    # nohup python -u int_sto.py atan10 $L $Ksto $M > Logs/TimeDep/out_atan5_${L}_${Ksto}_${M}.log 2>&1 &
    # sleep 2
    nohup python -u int_sto.py atan5 $L $Ksto $M > Logs/TimeDep/out_atan5_${L}_${Ksto}_${M}.log 2>&1 &
    sleep 2
    nohup python -u int_sto.py 0 $L $Ksto $M > Logs/TimeDep/out_0_${L}_${Ksto}_${M}.log 2>&1 &
    sleep 2
    
    # Columnar (requires a much higher K to see saturation)
    # nohup python -u int_col.py atan10 $L $Kcol $M  > Logs/Columnar/out_atan5_${L}_${Kcol}_${M}.log 2>&1 &
    # sleep 2
    nohup python -u int_col.py atan5 $L $Kcol $M  > Logs/Columnar/out_atan5_${L}_${Kcol}_${M}.log 2>&1 &
    sleep 2
    nohup python -u int_col.py 0 $L $Kcol $M  > Logs/Columnar/out_0_${L}_${Kcol}_${M}.log 2>&1 &
    sleep 2
    
done


echo "All simulations launched. Use 'ps aux | grep python' to monitor."


#! /bin/bash

###########################################################################
## Command explanation

# python -u: for returning output to log file in real time (unbuffered)
# int_sto.py/int_col.py: script name
# 1st argument: deltaname (0 or atan5)
# 2nd argument: L (system size)
# 3rd argument: K (coupling constant)
# 4th argument: M (number of seeds sampled)
###########################################################################

source ../scientific_kernel/bin/activate   


L=1000
Ksto=1           # 2024 paper
Kcol=40          # 2023 paper
M=400


## Large coupling

echo "## Running simulations. Date: `date`"
echo "Parameters: $L $Ksto $Kcol $M" 

# Time dependent
nohup python -u int_sto.py atan5 $L $Ksto $M > Logs/TimeDep/out_atan5_${L}_${Ksto}_${M}.log 2>&1 &
sleep 120
# nohup python -u int_sto.py 0 $L $Ksto $M > Logs/TimeDep/out_0_${L}_${Ksto}_${M}.log 2>&1 &

# Columnar (requires a much higher K to see saturation)
# nohup python -u int_col.py atan5 $L $Kcol $M  > Logs/Columnar/out_atan5_${L}_${Kcol}_${M}.log 2>&1 &
# sleep 120
# nohup python -u int_col.py 0 $L $Kcol $M  > Logs/Columnar/out_0_${L}_${Kcol}_${M}.log 2>&1 &


## Small coupling

# Time dependent
# nohup python -u int_sto.py 0 250 0.01 500 > Logs/out_sto_0_Ksmall.log 2>&1 &
# nohup python -u int_sto.py atan5 250 0.01 500  > Logs/out_sto_atan5_Ksmall.log 2>&1 &

# # Columnar
# nohup python -u int_col.py 0 250 0.02 500 > Logs/out_col_0_Ksmall.log 2>&1 &
# nohup python -u int_col.py atan5 250 0.02 500 > Logs/out_col_atan5_Ksmall.log 2>&1 &

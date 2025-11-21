#! /bin/bash


## Command explanation:
# python -u: for returning output to log file in real time (unbuffered)
# int_sto.py/int_col.py: script name
# 1st argument: deltaname (0 or atan5)
# 2nd argument: L (system size)
# 3rd argument: K (coupling constant)
# 4th argument: M (number of seeds sampled)


## Large coupling (K)

# Time dependent
# nohup python -u int_sto.py 0 500 1 500 > Logs/out_sto_0.log 2>&1 &
# nohup python -u int_sto.py atan5 500 1 500 > Logs/out_sto_atan5.log 2>&1 &

# # Columnar
nohup python -u int_col.py 0 500 5 500  > Logs/out_col_0.log 2>&1 &
nohup python -u int_col.py atan5 500 5 500  > Logs/out_col_atan5.log 2>&1 &


## Small coupling (K)

# Time dependent
# nohup python -u int_sto.py 0 500 0.01 500 > Logs/out_sto_0_Ksmall.log 2>&1 &
# nohup python -u int_sto.py atan5 500 0.01 500  > Logs/out_sto_atan5_Ksmall.log 2>&1 &

# # Columnar
# nohup python -u int_col.py 0 500 0.02 500 > Logs/out_col_0_Ksmall.log 2>&1 &
# nohup python -u int_col.py atan5 500 0.02 500 > Logs/out_col_atan5_Ksmall.log 2>&1 &
import numpy as np 
import analyze_data as adata
import matplotlib.pyplot as plot
import json
import time 
import constants as c
import fns
import sys



############################# User input: "RUN" option to run new simulation, "DATA" to execute data analysis code
#############################             "TESTING" to execute any testing code. If no option, same as "RUN"
n = len(sys.argv)
TESTING = 0
if(n != 1):
    if(sys.argv[1] == "RUN"):
        RUN_SIMUL = 1
        ANALYZE = 0
        TESTING = 0
    elif(sys.argv[1] == "DATA"):
        RUN_SIMUL = 0
        ANALYZE = 1
        TESTING = 0
    elif(sys.argv[1] == "TEST"):
        RUN_SIMUL = 0
        ANALYZE = 0
        TESTING = 1
else:
    RUN_SIMUL = 1
    ANALYZE = 0
    TESTING = 0

    
#############################
if RUN_SIMUL == 1:
    start_time = time.perf_counter()
    fns.main_iter_loop()
    end_time = time.perf_counter()
    print(f"Took a total time of {end_time-start_time} seconds")

if ANALYZE == 1:
    data_array = adata.read_data()
    #adata.summary_stats(data_array)
    adata.loop_visualization(data_array)
    #adata.harmonic_visualization(data_array)
    adata.slowmanifold_visualization(data_array)

if TESTING == 1: ## [Will implement convergence test. Will need to use multiple time steps.]
    data_array = adata.read_data()
    ##testing code here
   
    
















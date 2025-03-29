import os
import multiprocessing

# Get total CPU cores available to the process
num_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else multiprocessing.cpu_count()

print(f"Available CPUs: {num_cpus}")

num_cpus = int(os.getenv("SLURM_CPUS_ON_NODE", multiprocessing.cpu_count()))
print(f"Allocated CPUs: {num_cpus}")


import xgboost as xgb

num_cpus = min(multiprocessing.cpu_count(), len(os.sched_getaffinity(0)))
print(f"num_cpus = {num_cpus}")
xgb_model = xgb.XGBRegressor(n_jobs=num_cpus)

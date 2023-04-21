conda activate ebench

# CUDA_VISIBLE_DEVICES=1 python pytracking/run_experiment.py myexperiments esotVH_offline

# CUDA_VISIBLE_DEVICES=1 python pytracking/run_experiment.py myexperiments esotVM_offline

# CUDA_VISIBLE_DEVICES=1 python pytracking/run_experiment.py myexperiments esotVL_offline

CUDA_VISIBLE_DEVICES=0 python pytracking/run_experiment.py myexperiments esot500_fps_window_fe > log/esot500_fps_window_fe.log &
CUDA_VISIBLE_DEVICES=1 python pytracking/run_experiment.py myexperiments esot2_interp_fe


# free_mem = $(nvidia-smi --query-gpu=memory.free --format=csv -i 2 | grep -Eo [0-9]+)
# while [$free_mem -lt 10000 ]; do
#     free_mem = $(nvidia-smi --query-gpu=memory.free --format=csv -i 2 | grep -Eo [0-9]+)
#     sleep 10
# done

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
nohup python -u main.py > log_train 2>&1 &
tail -f log_train
 



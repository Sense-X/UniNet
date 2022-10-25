work_path=$(dirname $0)
filename=$(basename $work_path)
partition=$1
gpus=$2
OMP_NUM_THREADS=1 \
srun -p ${partition} -n ${gpus} --ntasks-per-node=8 --cpus-per-task=16 --gres=gpu:8 \
python -u main.py \
  --model UniNetB1 \
  --input-size 224 \
  --batch-size 128 \
  --output_dir ${work_path}/ckpt \
  --epochs 300 \
  --dist-eval \
  --drop-path 0.0 \
  --reprob 0.0 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --num_workers 8 \
  --port 29522 \
  --resume ${work_path}/ckpt/checkpoint.pth

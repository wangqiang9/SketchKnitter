NUM_GPUS=1
CUDA_VISIBLE_DEVICES=0 mpiexec -n $NUM_GPUS python train.py \
                --data_dir [/path/to/data] \
                --lr 1e-4 \
                --batch_size 4 \
                --use_fp16 False \
                --log_dir ./logs \
                --save_interval 100
                --diffusion_steps 100 \
                --noise_schedule linear \
                --image_size 96 \
                --num_channels 96 \
                --num_res_blocks 3

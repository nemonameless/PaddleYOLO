export FLAGS_allocator_strategy=auto_growth
model_type=yolov5u
job_name=yolov5u_s_300e_coco
job_name=yolov5u_n_300e_coco
# job_name=yolov5u_s_60e_voc
# job_name=yolov5u_l_60e_voc

config=configs/yolov5/${model_type}/${job_name}.yml
log_dir=log_dir/${job_name}
weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=2 python3.7 tools/train.py -c ${config} --eval #-r ${weights} --eval
python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval #-r ${weights}
#python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3 tools/train.py -c ${config} --eval #-r ${weights}

# 2. eval
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=2 python3.7 tools/eval.py -c ${config} -o weights=${weights}

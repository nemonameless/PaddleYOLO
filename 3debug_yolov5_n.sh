export FLAGS_allocator_strategy=auto_growth
name=n
model_type=yolov5
job_name=yolov5_${name}_300e_coco
config=configs/${model_type}/${job_name}.yml
log_dir=log_dir/${job_name}
weight_path=/paddle/yolo/follow/new_${name}.pdparams #../efficientteacher/efficient-yolov5${name}-ssod.pdparams
#weight_path=output/yolov5_n_300e_coco/9.pdparams #../efficientteacher/efficient-yolov5${name}-ssod.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=2 python3.7 tools/train.py -c ${config} --eval -r ${weight_path}
python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval #-r output/yolov5_n_300e_coco/9 #--eval
# &> ${job_name}.log &

# 2. eval
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=2 python3.7 tools/eval.py -c ${config} -o weights=${weight_path}

# 3. tools infer
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/infer.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams --infer_img=demo/000000014439.jpg
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/infer.py -c ${config} -o weights=${weight_path} --infer_img=demo/000000014439.jpg

# 4.export model
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/export_model.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/export_model.py -c ${config} -o weights=${weight_path} 

# 5. deploy infer
#CUDA_VISIBLE_DEVICES=0 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --video_file=test.mp4 --device=GPU

# 6. deploy speed
#CUDA_VISIBLE_DEVICES=0 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000570688.jpg --device=GPU --run_benchmark=True #--trt_max_shape=640 --trt_min_shape=640 --trt_opt_shape=640 --run_mode=trt_fp16

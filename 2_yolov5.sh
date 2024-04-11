export FLAGS_allocator_strategy=auto_growth
model_type=yolov5
job_name=yolov5_l_300e_coco
config=configs/${model_type}/${job_name}.yml
#config=configs/${model_type}/${job_name}_rect_eval.yml

log_dir=log_dir/${job_name}
weights=./output/${job_name}/9.pdparams


# 1. training
#CUDA_VISIBLE_DEVICES=2 python tools/train.py -c ${config} --eval #-r ${weights} --eval
python -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval #-r ${weights}
#python -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3 tools/train.py -c ${config} --eval #-r ${weights}

# 2. eval
#CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=1 python tools/eval.py -c ${config} -o weights=${weights}

# # 3.预测 (单张图/图片文件夹）
# CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg --draw_threshold=0.5
# # CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c ${config} -o weights=${weights} --infer_dir=demo/ --draw_threshold=0.5

# # 4.导出模型，以下3种模式选一种
# ## 普通导出，加trt表示用于trt加速，对NMS和silu激活函数提速明显
# CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c ${config} -o weights=${weights} # trt=True

# ## exclude_post_process去除后处理导出，返回和YOLOv5导出ONNX时相同格式的concat后的1个Tensor，是未缩放回原图的坐标+分类置信度
# # CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c ${config} -o weights=${weights} exclude_post_process=True # trt=True

# ## exclude_nms去除NMS导出，返回2个Tensor，是缩放回原图后的坐标和分类置信度
# # CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c ${config} -o weights=${weights} exclude_nms=True # trt=True

# # 5.部署预测，注意不能使用 去除后处理 或 去除NMS 导出后的模型去预测
# CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU

# # # 6.部署测速，加 “--run_mode=trt_fp16” 表示在TensorRT FP16模式下测速，注意如需用到 trt_fp16 则必须为加 trt=True 导出的模型
# # CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU --run_benchmark=True # --run_mode=trt_fp16

# YOLOv10

## 内容

- [模型库](#模型库)
- [使用教程](#使用教程)
- [引用](#引用)

## 模型库

### 基础检测模型

| 网络网络       | 输入尺寸 | 图片数/GPU | 学习率策略 | TRT-FP16-Latency(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |                                      下载链接                                      |               配置文件                |
|:-----------|:----:|:-------:|:-----:|:--------------------:|:-----------------------:|:------------------:|:---------:|:--------:|:------------------------------------------------------------------------------:|:---------------------------------:|
| *YOLOv10-n | 640  |   32    | 500e  |          -           |          38.5           |        53.9        |    2.3    |   6.7    | [下载链接](https://bj.bcebos.com/v1/paddledet/models/yolov10_n_500e_coco.pdparams) | [配置文件](./yolov10_n_500e_coco.yml) |
| *YOLOv10-s | 640  |   32    | 500e  |          -           |          46.3           |        63.0        |    7.2    |   21.6   | [下载链接](https://bj.bcebos.com/v1/paddledet/models/yolov10_s_500e_coco.pdparams) | [配置文件](./yolov10_s_500e_coco.yml) |
| *YOLOv10-m | 640  |   32    | 500e  |          -           |          51.1           |        68.2        |   15.4    |   59.1   | [下载链接](https://bj.bcebos.com/v1/paddledet/models/yolov10_m_500e_coco.pdparams) | [配置文件](./yolov10_m_500e_coco.yml) |
| *YOLOv10-b | 640  |   32    | 500e  |          -           |          52.5           |        69.6        |   19.1    |   92.0   | [下载链接](https://bj.bcebos.com/v1/paddledet/models/yolov10_b_500e_coco.pdparams) | [配置文件](./yolov10_b_500e_coco.yml) |
| *YOLOv10-x | 640  |   32    | 500e  |          -           |          54.4           |        71.3        |   29.5    |  160.4   | [下载链接](https://bj.bcebos.com/v1/paddledet/models/yolov10_x_500e_coco.pdparams) | [配置文件](./yolov10_x_500e_coco.yml) |

**注意:**

- YOLOv10模型mAP为部署权重在COCO val2017上的`mAP(IoU=0.5:0.95)`结果；
- YOLOv10模型训练使用COCO train2017作为训练集，Box AP为在COCO val2017上的`mAP(IoU=0.5:0.95)`结果；
- YOLOv10模型训练过程中默认使用8 GPUs进行混合精度训练，默认lr为0.01为8卡总batch_size的设置，如果**GPU卡数**或者每卡**batch
  size**发生改动，也不需要改动学习率，但为了保证高精度最好使用**总batch size大于64**的配置去训练；
- 如果你设置了`--run_benchmark=True`, 你首先需要安装以下依赖`pip install pynvml psutil GPUtil`。
- 默认在后处理部分开启`multi_label`，可以通过设置`-o multi_label=False`关闭
- 默认在验证集上的数据预处理部分使用`YOLOv5KeepRatioResize`和`LetterResize`来取得更高的mAP,
  详见[yolov10预处理配置文件](./_base_/yolov10_reader.yml)。

## 使用教程

### 0. **一键运行全流程**

将以下命令写在一个脚本文件里如```run.sh```，一键运行命令为：```sh run.sh```，也可命令行一句句去运行。

```bash
model_name=yolov10 # 可修改，如 ppyoloe
job_name=yolov10_s_500e_coco # 可修改，如 ppyoloe_plus_crn_s_80e_coco

config=configs/${model_name}/${job_name}.yml
log_dir=log_dir/${job_name}
# weights=https://bj.bcebos.com/v1/paddledet/models/${job_name}.pdparams
weights=output/${job_name}/model_final.pdparams

# 1.训练（单卡/多卡），加 --eval 表示边训边评估，加 --amp 表示混合精度训练
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c ${config} --eval --amp
python -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp

# 2.评估，加 --classwise 表示输出每一类mAP
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c ${config} -o weights=${weights} --classwise

# 3.预测 (单张图/图片文件夹）
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg --draw_threshold=0.5
# CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c ${config} -o weights=${weights} --infer_dir=demo/ --draw_threshold=0.5

# 4.导出模型，以下3种模式选一种
## 普通导出，加trt表示用于trt加速，对NMS和silu激活函数提速明显
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c ${config} -o weights=${weights} # trt=True

## exclude_post_process去除后处理导出，返回和YOLOv5导出ONNX时相同格式的concat后的1个Tensor，是未缩放回原图的坐标+分类置信度
# CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c ${config} -o weights=${weights} exclude_post_process=True # trt=True

# 5.部署预测，注意不能使用去除后处理导出后的模型去预测
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU

# 6.部署测速，加 “--run_mode=trt_fp16” 表示在TensorRT FP16模式下测速，注意如需用到 trt_fp16 则必须为加 trt=True 导出的模型
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU --run_benchmark=True # --run_mode=trt_fp16

# 7.onnx导出，一般结合 exclude_post_process去除后处理导出的模型
paddle2onnx --model_dir output_inference/${job_name} --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 12 --save_file ${job_name}.onnx

# 8.onnx trt测速
trtexec --onnx=${job_name}.onnx --workspace=4096 --avgRuns=10 --shapes=input:1x3x640x640 --fp16
trtexec --onnx=${job_name}.onnx --workspace=4096 --avgRuns=10 --shapes=input:1x3x640x640 --fp32
```

### 1. 训练

执行以下指令使用混合精度训练YOLOv10

```bash
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/yolov10/yolov10_s_500e_coco.yml --amp --eval
```

**注意:**

- `--amp`表示开启混合精度训练以避免显存溢出，`--eval`表示边训边验证。

### 2. 评估

执行以下命令在单个GPU上评估COCO val2017数据集

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/yolov10/yolov10_s_500e_coco.yml -o weights=https://bj.bcebos.com/v1/paddledet/models/yolov10_s_500e_coco.pdparams
```

### 3. 推理

使用以下命令在单张GPU上预测图片，使用`--infer_img`推理单张图片以及使用`--infer_dir`推理文件中的所有图片。

```bash
# 推理单张图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/yolov10/yolov10_s_500e_coco.yml -o weights=https://bj.bcebos.com/v1/paddledet/models/yolov10_s_500e_coco.pdparams --infer_img=demo/000000014439_640x640.jpg

# 推理文件中的所有图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/yolov10/yolov10_s_500e_coco.yml -o weights=https://bj.bcebos.com/v1/paddledet/models/yolov10_s_500e_coco.pdparams --infer_dir=demo
```

### 4.导出模型

YOLOv10在GPU上推理部署或benchmark测速等需要通过`tools/export_model.py`导出模型。

当你**使用Paddle Inference但不使用TensorRT**时，运行以下的命令导出模型

```bash
python tools/export_model.py -c configs/yolov10/yolov10_s_500e_coco.yml -o weights=https://bj.bcebos.com/v1/paddledet/models/yolov10_s_500e_coco.pdparams
```

当你**使用Paddle Inference且使用TensorRT**时，需要指定`-o trt=True`来导出模型。

```bash
python tools/export_model.py -c configs/yolov10/yolov10_s_500e_coco.yml -o weights=https://bj.bcebos.com/v1/paddledet/models/yolov10_s_500e_coco.pdparams trt=True
```

如果你想将YOLOv10模型导出为**ONNX格式**，参考
[PaddleDetection模型导出为ONNX格式教程](../../deploy/EXPORT_ONNX_MODEL.md)，运行以下命令：

```bash
# 导出推理模型
python tools/export_model.py -c configs/yolov10/yolov10_s_500e_coco.yml --output_dir=output_inference -o weights=https://bj.bcebos.com/v1/paddledet/models/yolov10_s_500e_coco.pdparams

# 安装paddle2onnx
pip install paddle2onnx

# 转换成onnx格式
paddle2onnx --model_dir output_inference/yolov10_s_500e_coco --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 11 --save_file yolov10_s_500e_coco.onnx
```

**注意：** ONNX模型目前只支持batch_size=1

## 引用

```
@article{wang2024yolov10,
  title={YOLOv10: Real-Time End-to-End Object Detection},
  author={Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2405.14458},
  year={2024}
}
```

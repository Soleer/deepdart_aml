#!/bin/bash
conda activate yolov5
python train.py --cfg yolov5s_custom.yaml --data dataset/keypoint.yaml --weights yolov5s.pt --batch-size 64 --epochs 10 --imgsz 800 
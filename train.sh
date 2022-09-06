#!/bin/bash
conda activate yolov5
python train.py --cfg yolov5s_custom.yaml --data keypoint.yaml --weights yolov5s.pt --workers 0 --epochs 10 --imgsz 800 --cache ram --batch 16
python val.py 
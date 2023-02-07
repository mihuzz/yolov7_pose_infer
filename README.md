# yolov7_pose_infer
>Based on Yolov7 and https://github.com/nanmi/yolov7-pose  
```python
"from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n-seg.pt")  # load an official model
model = YOLO("/home/mih/PycharmProjects/yomy8/venv/lib/python3.8/site-packages/ultralytics/weights4seg/yolov8n-seg.pt")  # load a custom trained

# Export the model
model.export(format="engine")"

```


>Convert onnx to trt:  
>/usr/local/TensorRT-8.2.5.1/bin/trtexec --onnx=/way/to/your/yolov7-w6-pose-sim-yolo.onnx --saveEngine=/path/to/save/yolov7-w6-pose-sim-yolo-fp16.engine --plugins=/path/where/is/libyolo.so    

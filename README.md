# yolov7_pose_infer
>Based on Yolov7 and https://github.com/nanmi/yolov7-pose   

>Convert to onnx:   
```python
import sys
sys.path.append('./')  # to run '$ python *.py' files in subdirectories
import torch
import torch.nn as nn
import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU

# Load PyTorch model
weights = './weights/yolov7-w6-pose.pt'
device = torch.device('cuda:0')
model = attempt_load(weights, map_location=device)  # load FP32 model

# Update model
for k, m in model.named_modules():
    m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    if isinstance(m, models.common.Conv):  # assign export-friendly activations
        if isinstance(m.act, nn.Hardswish):
            m.act = Hardswish()
        elif isinstance(m.act, nn.SiLU):
            m.act = SiLU()
model.model[-1].export = True # set Detect() layer grid export
model.eval()

# Input
img = torch.randn(1, 3, 960, 960).to(device)  # image size(1,3,320,192) iDetection
torch.onnx.export(model, img, './weights/yolov7-w6-pose.onnx', verbose=False, opset_version=12, input_names=['images'])

```
Use script to modify onnx model use netron to see digits of nodes:  
```
../YoloLayer_TRT_v7.0/add_custom_yolo_op.py
```
Build yolo layer tensorrt plugin  
```
cd {this repo}/YoloLayer_TRT_v7.0  
mkdir build && cd build  
cmake .. && make  
```
>Convert onnx to trt engine:  
```
cd {this repo}/

/usr/local/TensorRT-8.2.5.1/bin/trtexec --onnx=/way/to/your/yolov7-w6-pose-sim-yolo.onnx --saveEngine=/path/to/save/yolov7-w6-pose-sim-yolo-fp16.engine --plugins=/path/where/is/libyolo.so    
```

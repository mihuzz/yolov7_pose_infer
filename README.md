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
```python
import onnx_graphsurgeon as gs
import numpy as np
import onnx


# gs load a graph
graph = gs.import_onnx(onnx.load("./weights/yolov7-w6-pose.onnx"))

# Since we already know the names of the tensors we're interested in, we can
# grab them directly from the tensor map.
#
# NOTE: If you do not know the tensor names you want, you can view the graph in
# Netron to determine them, or use ONNX GraphSurgeon in an interactive shell
# to print the graph.
tensors = graph.tensors()

# If you want to embed shape information, but cannot use ONNX shape inference,
# you can manually modify the tensors at this point:
#
# IMPORTANT: You must include type information for input and output tensors if it is not already
# present in the graph.
#
# NOTE: ONNX GraphSurgeon will also accept dynamic shapes - simply set the corresponding
# dimension(s) to `gs.Tensor.DYNAMIC`, e.g. `shape=(gs.Tensor.DYNAMIC, 3, 224, 224)`
inputs = [tensors["737"].to_variable(dtype=np.float32),
    tensors["775"].to_variable(dtype=np.float32),
    tensors["813"].to_variable(dtype=np.float32),
    tensors["851"].to_variable(dtype=np.float32)]

# Add a output tensor of new graph
modified_output = gs.Variable(name="output0", dtype=np.float32, shape=(57001, 1, 1))

# Add a new node that you want
new_node = gs.Node(op="YoloLayer_TRT", name="YoloLayer_TRT_0", inputs=inputs, outputs=[modified_output])

# append into graph
graph.nodes.append(new_node)
graph.outputs = [modified_output]

graph.cleanup().toposort()

# gs save a graph
onnx.save(gs.export_onnx(graph), "./weights/yolov7-w6-pose-sim-yolo.onnx")
```


>Convert onnx to trt:  
>/usr/local/TensorRT-8.2.5.1/bin/trtexec --onnx=/way/to/your/yolov7-w6-pose-sim-yolo.onnx --saveEngine=/path/to/save/yolov7-w6-pose-sim-yolo-fp16.engine --plugins=/path/where/is/libyolo.so    

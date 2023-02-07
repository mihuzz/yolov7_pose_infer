# yolov7_pose_infer
>Based on Yolov7 and https://github.com/nanmi/yolov7-pose  

Convert to trt:  
>/usr/local/TensorRT-8.2.5.1/bin/trtexec --onnx=/way/to/your/yolov7-w6-pose-sim-yolo.onnx --saveEngine=/path/to/save/yolov7-w6-pose-sim-yolo-fp16.engine --plugins=/path/where/is/libyolo.so    

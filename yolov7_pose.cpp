// based on https://github.com/nanmi/yolov7-pose
#include <map>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <NvInfer.h>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "utils.h"


using namespace cv;
using namespace Yolo;



int main(int argc, char** argv) {


    cv::CommandLineParser parser(argc, argv, keys);

    const std::string modeleFile = parser.get<String>("model");

    parser = CommandLineParser(argc, argv, keys);
    parser.about("Use this script to run Yolov7 segmantation using OpenCV.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }


    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file(modeleFile, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    else {
        std::cout << "load engine failed" << std::endl;
        return 1;
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;


    int input_size = 1*3*960*960;
    int output_size = 57001 * 1*1;
    static float* prob = new float[output_size];


    cv::Mat frame;

    // Open a video file or an image file or a camera stream.
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<String>("input"));
    else
        cap.open(parser.get<int>("device"), cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    std::string outfile = "/path/to/save/yolov7pose.avi";
    cv::VideoWriter myvideo(outfile, cv::VideoWriter::fourcc('I','4','2','0'), 29, cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

    for(;;)
    {
        cap >> frame;

        if(frame.empty()){
            std::cout << "Can't read frame or end of file " << std::endl;
            break;

        }
            cv::Mat pr_img = static_resize(frame);
    
            float* blob;
            blob = blobFromImage(pr_img);

            // run inference
            auto start = std::chrono::system_clock::now();
            doInference(*context, blob, prob, output_size, input_size);
            auto end = std::chrono::system_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

            // decode
            std::map<float, std::vector<Detection>> objects_map;
            postprocess_decode(prob, CONF_THRESH, objects_map);

            // NMS
            std::vector<Detection> objects;
            nms(objects_map, objects, NMS_THRESH);

            std::cout << "NMS quantities: " << objects.size() << std::endl;

            for (size_t j = 0; j < objects.size(); j++) {
                cv::Rect obj_bbox = get_bbox(frame, objects[j].bbox);

                std::cout << "x: " << objects[j].bbox[0] << "  y:" << objects[j].bbox[1] << "   w:" << objects[j].bbox[2] << "  h:" << objects[j].bbox[3] << std::endl;
                cv::rectangle(frame, obj_bbox, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(frame, categories[(int)(objects[j].class_id)] + std::to_string(objects[j].conf), cv::Point(obj_bbox.x, obj_bbox.y - 1), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(200, 20, 255), 1.7);

                get_kpts(frame, objects[j].kpts);
                for (size_t i = 0; i < KEY_POINTS_NUM; i++)
                {
                    const unsigned char* kpt_color = pose_kpt_color[i];
                    cv::circle(frame, cv::Point((int)(objects[j].kpts[i].x), (int)(objects[j].kpts[i].y)), 2, cv::Scalar(kpt_color[0], kpt_color[1], kpt_color[2]), -1);
                }
                for (size_t i = 0; i < 19; i++)
                {
                    int x1 = (int)(objects[j].kpts[skeleton[i][0]-1].x);
                    int y1 = (int)(objects[j].kpts[skeleton[i][0]-1].y);
                    int x2 = (int)(objects[j].kpts[skeleton[i][1]-1].x);
                    int y2 = (int)(objects[j].kpts[skeleton[i][1]-1].y);

                    const unsigned char* limb_color = pose_limb_color[i];
                    cv::line(frame, cv::Point(x1,y1), cv::Point(x2,y2), cv::Scalar(limb_color[0], limb_color[1], limb_color[2]), 2);
                }

            }
            myvideo.write(frame);

            cv::imshow("result", frame);
            if (cv::waitKey(1)==27)
                        break;


    delete blob;
    // destroy the engine
    cudaFreeHost(context);
    cudaFreeHost(engine);
    cudaFreeHost(runtime);
    }

    return 0;
}

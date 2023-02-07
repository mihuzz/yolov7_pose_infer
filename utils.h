#ifndef TRTX_YOLOV5_UTILS_H_
#define TRTX_YOLOV5_UTILS_H_

#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include "logging.h"
#include "yololayer.h"

using namespace Yolo;
using namespace nvinfer1;


std::string keys =
    "{ help  h     | | Print help message. }"
    "{ model       | | Path to input model. }"
    "{ device      |  0 | camera device number. }"
    "{ input      | | Path to input image or video file. Skip this argument to capture frames from a camera. }";


#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.65
#define CONF_THRESH 0.75


std::string categories[] = {"person"};
static const unsigned char pose_kpt_color[17][3] =
{
    {  0, 255,   0},
    {  0, 255,   0},
    {  0, 255,   0},
    {  0, 255,   0},
    {  0, 255,   0},
    {255, 128,   0},
    {255, 128,   0},
    {255, 128,   0},
    {255, 128,   0},
    {255, 128,   0},
    {255, 128,   0},
    { 51, 153, 255},
    { 51, 153, 255},
    { 51, 153, 255},
    { 51, 153, 255},
    { 51, 153, 255},
    { 51, 153, 255}
};
static const unsigned char pose_limb_color[19][3] =
{
    { 51, 153, 255},
    { 51, 153, 255},
    { 51, 153, 255},
    { 51, 153, 255},
    {255,  51, 255},
    {255,  51, 255},
    {255,  51, 255},
    {255, 128,   0},
    {255, 128,   0},
    {255, 128,   0},
    {255, 128,   0},
    {255, 128,   0},
    {  0, 255,   0},
    {  0, 255,   0},
    {  0, 255,   0},
    {  0, 255,   0},
    {  0, 255,   0},
    {  0, 255,   0},
    {  0, 255,   0}
};
static const unsigned char skeleton[19][2] =
{
    {16, 14},
    {14, 12},
    {17, 15},
    {15, 13},
    {12, 13},
    {6,  12},
    {7,  13},
    {6,   7},
    {6,   8},
    {7,   9},
    {8,  10},
    {9,  11},
    {2,   3},
    {1,   2},
    {1,   3},
    {2,   4},
    {3,   5},
    {4,   6},
    {5,   7}
};
// stuff we know about the network and the input/output blobs
const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME = "output0";
static Logger gLogger;


float* blobFromImage(cv::Mat& img){
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (int c = 0; c < channels; c++)
    {
        for (int  h = 0; h < img_h; h++)
        {
            for (int w = 0; w < img_w; w++)
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return blob;
}

cv::Mat static_resize(cv::Mat& img) {
    float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(INPUT_W, INPUT_H, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
//    std::cout << "out.cols :" << out.rows << "\n";
    return out;
}


cv::Rect get_bbox(cv::Mat& img, float bbox[4]) {
    float l, r, t, b;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f;// - (INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f;// - (INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f;// - (INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f;// - (INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(round(l), round(t), round(r - l), round(b - t));
}

int get_kpts(cv::Mat& img, Keypoint kpts[17]) {
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    for (int i = 0; i < 17; i++)
    {
        if (r_h > r_w) {
            kpts[i].x /= r_w;
            kpts[i].y /= r_w;
        } else {
            kpts[i].x /= r_h;
            kpts[i].y /= r_h;
        }
    }

    return 0;
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool cmp(const Detection& a, const Detection& b) {
    return a.conf > b.conf;
}

void nms(std::map<float, std::vector<Detection>>& objects_map, std::vector<Detection>& res, float nms_thresh = 0.5) {
    for (auto it = objects_map.begin(); it != objects_map.end(); it++)
    {
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (unsigned int det_map_i = 0; det_map_i < dets.size(); ++det_map_i) {
            auto& item = dets[det_map_i];
            res.push_back(item);
            for (unsigned int n = det_map_i + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

static void postprocess_decode(float* feat_blob, float prob_threshold, std::map<float, std::vector<Detection>>& objects_map)
{
    int det_size = sizeof(Detection) / sizeof(float);
    for (int i = 0; i < feat_blob[0] && i < MAX_OUTPUT_BBOX_COUNT; i++) {
        if (feat_blob[1 + det_size * i + 4] <= prob_threshold) continue;
        Detection det;
        memcpy(&det, &feat_blob[1 + det_size * i], det_size * sizeof(float));
        if (objects_map.count(det.class_id) == 0) objects_map.emplace(det.class_id, std::vector<Detection>());
        objects_map[det.class_id].push_back(det);
    }
}



void doInference(IExecutionContext& context, float* input, float* output, const int output_size, const int input_shape) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    // int mBatchSize = engine.getMaxBatchSize();

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], input_shape * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, input_shape * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

#endif  // TRTX_YOLOV5_UTILS_H_


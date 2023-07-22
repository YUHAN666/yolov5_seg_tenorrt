#include "onnx2engine.h"
#include "preprocess.h"
#include "postprocess.h"
#include "cuda_utils.h"
#include "get_image_path.h"
#include <string>



using namespace std;
using namespace nvinfer1;


extern const int kBatchSize = 1;
extern const int classNum = 1;
extern const int kInputH = 640;
extern const int kInputW = 640;
const static int kMaxNumOutputBbox = 25200;
const static int kOutputSize1 = 32 * (kInputH / 4) * (kInputW / 4); 
const static int kOutputSize2 = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float);
const static float kNmsThresh = 0.45f;
const static float kConfThresh = 0.5f;

/// <summary>
///  为输入输出张量开辟CPU和GPU内存
/// </summary>
/// <param name="engine"></param>
/// <param name="gpu_input_buffer"></param>
/// <param name="gpu_output_buffer1"></param>
/// <param name="gpu_output_buffer2"></param>
/// <param name="cpu_output_buffer1"></param>
/// <param name="cpu_output_buffer2"></param>
void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer1, float** gpu_output_buffer2, float** cpu_output_buffer1, float** cpu_output_buffer2) {
    assert(engine->getNbBindings() == 3);

    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    // 1* 32 * 160 * 160
    CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer1, kBatchSize * kOutputSize1 * sizeof(float)));
    // 1* kMaxNumOutputBbox * (4 + 1 + 80 + 32)
    CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer2, kBatchSize * kOutputSize2 * sizeof(float)));

    // Alloc CPU buffers
    *cpu_output_buffer1 = new float[kBatchSize * kOutputSize1];
    *cpu_output_buffer2 = new float[kBatchSize * kOutputSize2];
}

/// <summary>
/// 执行推理并将结果从GPU拷贝至CPU内存
/// </summary>
/// <param name="context"></param>
/// <param name="stream"></param>
/// <param name="buffers"></param>
/// <param name="output1"></param>
/// <param name="output2"></param>
/// <param name="batchSize"></param>
void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output1, float* output2, int batchSize) {
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output1, buffers[1], batchSize * kOutputSize1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(output2, buffers[2], batchSize * kOutputSize2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}


void inference1(string engineFile, string labelFile, std::vector<std::string> imagePaths) {
    MyLogger logger;
    fstream file;
#pragma region 从文件读取Engine
    cout << "loading filename from:" << engineFile << endl;
    nvinfer1::IRuntime* trtRuntime;
    file.open(engineFile, ios::binary | ios::in);
    file.seekg(0, ios::end);
    int length = file.tellg();
    file.seekg(0, ios::beg);
    std::unique_ptr<char[]> data(new char[length]);
    file.read(data.get(), length);
    file.close();
    cout << "load engine done" << endl;
#pragma endregion

    // 创建runtime，并反序列化engine
    std::cout << "deserializing" << endl;
    trtRuntime = createInferRuntime(logger);
    ICudaEngine* engine = trtRuntime->deserializeCudaEngine(data.get(), length);
    cout << "deserialize done" << endl;

    // 创建执行环境
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // 看一下输入输出形状
    // input (1, 3, 640, 640)
    // output1 (1, 32, 160, 160)
    // output2 (1, 25200, 117)
    //for (int i = 0; i < engine->getNbBindings(); i++) {
    //    Dims dim = engine->getBindingDimensions(i);
    //    continue;
    //}

    // 读取label.txt
    std::unordered_map<int, std::string> labels_map;
    read_labels(labelFile, labels_map);

    // 创建cuda流
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    const static int kMaxInputImageSize = 4096 * 3112;
    cuda_preprocess_init(kMaxInputImageSize);

    // Prepare cpu and gpu buffers
    float* gpu_buffers[3];
    float* cpu_output_buffer1 = nullptr;
    float* cpu_output_buffer2 = nullptr;
    prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &gpu_buffers[2], &cpu_output_buffer1, &cpu_output_buffer2);

    int kBatchSize = 1;
    // batch predict
    for (size_t i = 0; i < imagePaths.size(); i += kBatchSize) {
        // Get a batch of images
        std::vector<cv::Mat> img_batch;
        std::vector<std::string> img_name_batch;
        for (size_t j = i; j < i + kBatchSize; j++) {
            cv::Mat img = cv::imread(imagePaths[i]);
            img_batch.push_back(img);
            img_name_batch.push_back(imagePaths[i]);
        }

        // Preprocess，将图片resize到640,640
        cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

        // Run inference
        auto start = std::chrono::system_clock::now();
        infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer1, cpu_output_buffer2, kBatchSize);


        // NMS
        std::vector<std::vector<Detection>> res_batch;
        batch_nms(res_batch, cpu_output_buffer2, img_batch.size(), kOutputSize2, kConfThresh, kNmsThresh);

        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        // Draw result and save image
        for (int b = 0; b < img_name_batch.size(); b++) {
            auto& res = res_batch[b];
            cv::Mat img = img_batch[b];

            auto masks = process_mask(&cpu_output_buffer1[b * kOutputSize1], kOutputSize1, res);
            draw_mask_bbox(img, res, masks, labels_map);

            string savPath = "./results/" +  std::to_string(i) + '_' + std::to_string(b) + ".jpg";

            cv::imwrite(savPath, img);
        }
    }
}


int main()
{
    string engineFile = "./1.engine";
    string labelFile = "./1.txt";
    std::string folderPath = "E:/CODES/YOLO V5/yolov5-master/datasets/pzt/images";
    std::vector<std::string> imagePaths = getImagePathsInFolder(folderPath);
    //GetEngine(engineFile);
    inference1(engineFile, labelFile, imagePaths);
    return true;
}

#include "onnx2engine.h"

void GetEngine(MyLogger logger) {

    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

    const uint32_t explicit_batch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicit_batch);

    const std::string onnx_model = "./yolov5s-seg.onnx";
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    parser->parseFromFile(onnx_model.c_str(),
        static_cast<int>(nvinfer1::ILogger::Severity::kERROR));
    // 如果有错误则输出错误信息
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
        std::cout << parser->getError(i)->desc() << std::endl;
    }


    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    //config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 25);
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }


    std::cout << "start building engine" << std::endl;
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build engine done" << std::endl;

    // 序列化引擎
    nvinfer1::IHostMemory* trtModelStream = engine->serialize();

    // 保存引擎
    nvinfer1::IHostMemory* data = engine->serialize();
    std::ofstream file;
    file.open("./last.engine", std::ios::binary | std::ios::out);
    std::cout << "writing engine file..." << std::endl;
    file.write((const char*)data->data(), data->size());
    std::cout << "save engine file done" << std::endl;
    file.close();

    delete config;
    delete parser;
    delete network;
    delete builder;

}
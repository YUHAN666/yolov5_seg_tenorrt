#pragma once
#include "NvInfer.h"
#include "NvOnnxParser.h"


class MyLogger : public nvinfer1::ILogger {
public:
    explicit MyLogger(nvinfer1::ILogger::Severity severity =
        nvinfer1::ILogger::Severity::kWARNING)
        : severity_(severity) {}

    void log(nvinfer1::ILogger::Severity severity,
        const char* msg) noexcept override {
        if (severity <= severity_) {
            std::cerr << msg << std::endl;
        }
    }
    nvinfer1::ILogger::Severity severity_;
};
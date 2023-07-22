#include <algorithm>
#include <numeric>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>


extern const int classNum;
extern const int kInputH;
extern const int kInputW;

struct alignas(float) Detection {
    float bbox[4];  // center_x center_y w h
    float conf;  // bbox_conf * cls_conf
    float class_id[1];
    float mask[32];
};


void batch_nms(std::vector<std::vector<Detection>>& res_batch, float* output, int batch_size, int output_size, float conf_thresh, float nms_thresh);

std::vector<cv::Mat> process_mask(const float* proto, int proto_size, std::vector<Detection>& dets);

void draw_mask_bbox(cv::Mat& img, std::vector<Detection>& dets, std::vector<cv::Mat>& masks, std::unordered_map<int, std::string>& labels_map);

// Function to trim leading and trailing whitespace from a string
static inline std::string trim_leading_whitespace(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}


static inline int read_labels(const std::string labels_filename, std::unordered_map<int, std::string>& labels_map) {

    std::ifstream file(labels_filename);
    // Read each line of the file
    std::string line;
    int index = 0;
    while (std::getline(file, line)) {
        // Strip the line of any leading or trailing whitespace
        line = trim_leading_whitespace(line);

        // Add the stripped line to the labels_map, using the loop index as the key
        labels_map[index] = line;
        index++;
    }
    // Close the file
    file.close();

    return 0;
}
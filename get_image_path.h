#include <iostream>
#include <windows.h>
#include <vector>
#include <string>
#include <algorithm>

// 判断文件是否为图片
bool isImageFile(const std::string& filename) {
    // 可以根据实际需求判断文件是否为图片，这里简单示范判断文件后缀名
    // 假设支持常见图片格式：.jpg, .jpeg, .png, .gif
    std::string extension = filename.substr(filename.find_last_of('.') + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    return (extension == "jpg" || extension == "jpeg" || extension == "png" || extension == "bmp");
}

// 获取指定文件夹下所有图片的路径
std::vector<std::string> getImagePathsInFolder(const std::string& folderPath) {
    std::vector<std::string> imagePaths;
    WIN32_FIND_DATA findFileData;
    HANDLE hFind = FindFirstFile((folderPath + "/*").c_str(), &findFileData);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                std::string filename = findFileData.cFileName;
                if (isImageFile(filename)) {
                    std::string fullPath = folderPath + "/" + filename;
                    imagePaths.push_back(fullPath);
                }
            }
        } while (FindNextFile(hFind, &findFileData) != 0);

        FindClose(hFind);
    }

    return imagePaths;
}
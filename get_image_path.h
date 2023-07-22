#include <iostream>
#include <windows.h>
#include <vector>
#include <string>
#include <algorithm>

// �ж��ļ��Ƿ�ΪͼƬ
bool isImageFile(const std::string& filename) {
    // ���Ը���ʵ�������ж��ļ��Ƿ�ΪͼƬ�������ʾ���ж��ļ���׺��
    // ����֧�ֳ���ͼƬ��ʽ��.jpg, .jpeg, .png, .gif
    std::string extension = filename.substr(filename.find_last_of('.') + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    return (extension == "jpg" || extension == "jpeg" || extension == "png" || extension == "bmp");
}

// ��ȡָ���ļ���������ͼƬ��·��
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
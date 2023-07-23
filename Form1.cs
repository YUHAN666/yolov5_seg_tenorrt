using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using OpenCvSharp;
using System.Runtime.InteropServices;
using System.Net.Http.Headers;

namespace tensorrt_deploy_csharp
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            ImageProcessor processor = new ImageProcessor();
            processor.ProcessImage("./1.bmp");
        }
    }


    public class ImageProcessor
    {
        [DllImport("CudaRuntime1.dll", CharSet = CharSet.Ansi)]
        public static extern void YourCppFunction(float[] image, int rows, int cols, int channels);

        [DllImport("CudaRuntime1.dll", CharSet = CharSet.Ansi)]
        public static extern void Inference(string engineFile, string labelFile, string folderPath);

        [DllImport("CudaRuntime1.dll", CharSet = CharSet.Ansi)]
        public static extern unsafe IntPtr YoloFactory(string engineFile, string labelFile);

        [DllImport("CudaRuntime1.dll", CharSet = CharSet.Ansi)]
        public static extern bool RunYolo(IntPtr handle, byte[] image, int rows, int cols, int channels);

        public void ProcessImage(string imagePath)
        {
            Mat image = Cv2.ImRead(imagePath, ImreadModes.Color);

            //image.ConvertTo(image, MatType.CV_32F);

            byte[] imageArray = new byte[image.Rows * image.Cols * image.Channels()];
            Marshal.Copy(image.Data, imageArray, 0, imageArray.Length);

            //YourCppFunction(imageArray, image.Rows, image.Cols, image.Channels());

            string engineFile = "./1.engine";
            string labelFile = "./1.txt";
            //string folderPath = "E:/CODES/YOLO V5/yolov5-master/datasets/pzt/images";

            //Inference(engineFile, labelFile, folderPath);
            IntPtr padd = YoloFactory(engineFile, labelFile);
            var result = RunYolo(padd, imageArray, image.Rows, image.Cols, image.Channels());

        }
    }

}

/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/xf_headers.hpp"
#include "xf_canny_config.h"

#include "xcl2.hpp"

typedef unsigned char NMSTYPE;

NMSTYPE Filter3x3(
    NMSTYPE t0, NMSTYPE t1, NMSTYPE t2, NMSTYPE m0, NMSTYPE m1, NMSTYPE m2, NMSTYPE b0, NMSTYPE b1, NMSTYPE b2) {
    NMSTYPE value = false;
    int g0 = t0 + t2 + b0 + b2;
    int g1 = (t1 + b1 + m0 + m2) << 1;
    int g2 = m1 << 2;

    value = ((int)(g0 + g1 + g2) >> 4);
    return value;
}
void AverageGaussian(cv::Mat& src, cv::Mat& dst) {
    int i, j;
    NMSTYPE t0, t1, t2;
    NMSTYPE m0, m1, m2;
    NMSTYPE b0, b1, b2;
    NMSTYPE result;

    /*			First row			*/
    i = 0;
    for (j = 0; j < src.cols; j++) {
        if (j == 0) {
            t0 = 0;
            t1 = 0;
            t2 = 0;
            m0 = 0;
            m1 = src.at<NMSTYPE>(i, j);
            m2 = src.at<NMSTYPE>(i, j + 1);
            b0 = 0;
            b1 = src.at<NMSTYPE>(i + 1, j);
            b2 = src.at<NMSTYPE>(i + 1, j + 1);
        } else if ((j > 0) && (j < src.cols - 1)) {
            t0 = 0;
            t1 = 0;
            t2 = 0;
            m0 = src.at<NMSTYPE>(i, j - 1);
            m1 = src.at<NMSTYPE>(i, j);
            m2 = src.at<NMSTYPE>(i, j + 1);
            b0 = src.at<NMSTYPE>(i + 1, j - 1);
            b1 = src.at<NMSTYPE>(i + 1, j);
            b2 = src.at<NMSTYPE>(i + 1, j + 1);
        } else if (j == src.cols - 1) {
            t0 = 0;
            t1 = 0;
            t2 = 0;
            m0 = src.at<NMSTYPE>(i, j - 1);
            m1 = src.at<NMSTYPE>(i, j);
            m2 = 0;
            b0 = src.at<NMSTYPE>(i + 1, j - 1);
            b1 = src.at<NMSTYPE>(i + 1, j);
            b2 = 0;
        }
        result = Filter3x3(t0, t1, t2, m0, m1, m2, b0, b1, b2);
        dst.at<uchar>(i, j) = result;
    }
    for (i = 1; i < src.rows - 1; i++) {
        for (j = 0; j < src.cols; j++) {
            if (j == 0) {
                t0 = 0;
                t1 = src.at<NMSTYPE>(i - 1, j);
                t2 = src.at<NMSTYPE>(i - 1, j + 1);
                m0 = 0;
                m1 = src.at<NMSTYPE>(i, j);
                m2 = src.at<NMSTYPE>(i, j + 1);
                b0 = 0;
                b1 = src.at<NMSTYPE>(i + 1, j);
                b2 = src.at<NMSTYPE>(i + 1, j + 1);
            } else if ((j > 0) && (j < src.cols - 1)) {
                t0 = src.at<NMSTYPE>(i - 1, j - 1);
                t1 = src.at<NMSTYPE>(i - 1, j);
                t2 = src.at<NMSTYPE>(i - 1, j + 1);
                m0 = src.at<NMSTYPE>(i, j - 1);
                m1 = src.at<NMSTYPE>(i, j);
                m2 = src.at<NMSTYPE>(i, j + 1);
                b0 = src.at<NMSTYPE>(i + 1, j - 1);
                b1 = src.at<NMSTYPE>(i + 1, j);
                b2 = src.at<NMSTYPE>(i + 1, j + 1);
            } else if (j == src.cols - 1) {
                t0 = src.at<NMSTYPE>(i - 1, j - 1);
                t1 = src.at<NMSTYPE>(i - 1, j);
                t2 = 0;
                m0 = src.at<NMSTYPE>(i, j - 1);
                m1 = src.at<NMSTYPE>(i, j);
                m2 = 0;
                b0 = src.at<NMSTYPE>(i + 1, j - 1);
                b1 = src.at<NMSTYPE>(i + 1, j);
                b2 = 0;
            }
            result = Filter3x3(t0, t1, t2, m0, m1, m2, b0, b1, b2);
            dst.at<uchar>(i, j) = result;
        }
    }
    /*			Last row			*/
    i = src.rows - 1;
    for (j = 0; j < src.cols; j++) {
        if (j == 0) {
            t0 = 0;
            t1 = src.at<NMSTYPE>(i - 1, j);
            t2 = src.at<NMSTYPE>(i - 1, j + 1);
            m0 = 0;
            m1 = src.at<NMSTYPE>(i, j);
            m2 = src.at<NMSTYPE>(i, j + 1);
            b0 = 0;
            b1 = 0; // src.at<NMSTYPE>(i+1, j);
            b2 = 0; // src.at<NMSTYPE>(i+1, j+1);
        } else if ((j > 0) && (j < src.cols - 1)) {
            t0 = src.at<NMSTYPE>(i - 1, j - 1);
            t1 = src.at<NMSTYPE>(i - 1, j);
            t2 = src.at<NMSTYPE>(i - 1, j + 1);
            m0 = src.at<NMSTYPE>(i, j - 1);
            m1 = src.at<NMSTYPE>(i, j);
            m2 = src.at<NMSTYPE>(i, j + 1);
            b0 = 0;
            b1 = 0;
            b2 = 0;
        } else if (j == src.cols - 1) {
            t0 = src.at<NMSTYPE>(i - 1, j - 1);
            t1 = src.at<NMSTYPE>(i - 1, j);
            t2 = 0;
            m0 = src.at<NMSTYPE>(i, j - 1);
            m1 = src.at<NMSTYPE>(i, j);
            m2 = 0;
            b0 = 0;
            b1 = 0;
            b2 = 0;
        }
        result = Filter3x3(t0, t1, t2, m0, m1, m2, b0, b1, b2);
        dst.at<uchar>(i, j) = result;
    }
}

int main(int argc, char** argv) {
    //# Images
    cv::Mat in_img_1, in_img_2;
    cv::Mat img_gray_1, img_gray_2, hls_img, ocv_img, out_img, out_img_edge;
    cv::Mat diff;

    if (argc != 3) {
        printf("Usage : <executable> <input image> \n");
        return -1;
    }

    char img_path[1000];

    in_img_1 = cv::imread(argv[1], 1); // reading in the color image
    in_img_2 = cv::imread(argv[2], 1); // reading in the color image
    if ((!in_img_1.data)||(!in_img_2.data))
	{
        printf("Failed to load the image ... %s\n!", argv[1]);
        return -1;
    }

    extractChannel(in_img_1, img_gray_1, 1); // Extract gray scale image
    extractChannel(in_img_2, img_gray_2, 1); // Extract gray scale image

    hls_img.create(img_gray_1.rows, img_gray_1.cols, img_gray_1.depth());      // HLS image creation
    out_img.create(img_gray_1.rows, img_gray_1.cols / 4, img_gray_1.depth());  // HLS image creation
    out_img_edge.create(img_gray_1.rows, img_gray_1.cols, img_gray_1.depth()); // HLS image creation

    int height, width;
    int low_threshold, high_threshold;
    height = img_gray_1.rows;
    width = img_gray_1.cols;
    low_threshold = 30;
    high_threshold = 64;

	#if FILTER_WIDTH == 3
		float sigma = 0.5f;
	#endif
	#if FILTER_WIDTH == 7
		float sigma = 1.16666f;
	#endif
	#if FILTER_WIDTH == 5
		float sigma = 0.8333f;
	#endif

    //////////////////////////////////////////////////////CL///////////////////////////////////

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device);

    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_canny");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);
    cl::Kernel krnl(program, "canny_accel");

    std::vector<cl::Memory> inBufVec, outBufVec;
    cl::Buffer imageToDevice_1(context, CL_MEM_READ_ONLY, (height * width));
    cl::Buffer imageToDevice_2(context, CL_MEM_READ_ONLY, (height * width));
    cl::Buffer imageFromDevice(context, CL_MEM_READ_WRITE, (height * width / 4));

    // Set the kernel arguments
    krnl.setArg(0, imageToDevice_1);
    krnl.setArg(1, imageToDevice_2);
    krnl.setArg(2, imageFromDevice);
    krnl.setArg(3, height);
    krnl.setArg(4, width);
    krnl.setArg(5, sigma);
    krnl.setArg(6, low_threshold);
    krnl.setArg(7, high_threshold);

    q.enqueueWriteBuffer(imageToDevice_1, CL_TRUE, 0, (height * (width)), img_gray_1.data);
    q.enqueueWriteBuffer(imageToDevice_2, CL_TRUE, 0, (height * (width)), img_gray_2.data);
    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    printf("before kernel");
    // Launch the kernel
    q.enqueueTask(krnl, NULL, &event_sp);
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("after kernel");

    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    // Copying Device result data to Host memory
    // q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, (height*width/4), out_img.data);
    // q.finish();

    cl::Kernel krnl2(program, "edgetracing_accel");
    // cl::Buffer imageToDeviceedge(context, CL_MEM_READ_WRITE,(height*width/4));
    cl::Buffer imageFromDeviceedge(context, CL_MEM_WRITE_ONLY, (height * width));

    // Set the kernel arguments
    krnl2.setArg(0, imageFromDevice);
    krnl2.setArg(1, imageFromDeviceedge);
    krnl2.setArg(2, height);
    krnl2.setArg(3, width);

    // q.enqueueWriteBuffer(imageToDeviceedge, CL_TRUE, 0, (height*(width/4)), out_img.data);

    // Profiling Objects
    cl_ulong startedge = 0;
    cl_ulong endedge = 0;
    double diff_prof_edge = 0.0f;
    cl::Event event_sp_edge;

    printf("before kernel");
    // Launch the kernel
    q.enqueueTask(krnl2, NULL, &event_sp_edge);
    clWaitForEvents(1, (const cl_event*)&event_sp_edge);

    printf("after kernel");

    event_sp_edge.getProfilingInfo(CL_PROFILING_COMMAND_START, &startedge);
    event_sp_edge.getProfilingInfo(CL_PROFILING_COMMAND_END, &endedge);
    diff_prof_edge = endedge - startedge;
    std::cout << (diff_prof_edge / 1000000) << "ms" << std::endl;

    // Copying Device result data to Host memory
    //q.enqueueReadBuffer(imageFromDeviceedge, CL_TRUE, 0, (height * width), out_img_edge.data);

    cl_int err;
    cv::Mat element = cv::getStructuringElement(KERNEL_SHAPE, cv::Size(FILTER_SIZE, FILTER_SIZE), cv::Point(-1, -1));

    unsigned char structure_element[FILTER_SIZE * FILTER_SIZE];

    for (int i = 0; i < (FILTER_SIZE * FILTER_SIZE); i++) {
        structure_element[i] = element.data[i];
    }

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "dilation_accel", &err));

    OCL_CHECK(err, cl::Buffer imageFromDeviceDilate(context, CL_MEM_WRITE_ONLY, (height * width * CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer kernelFilterToDevice(context, CL_MEM_READ_ONLY,
                                                   (FILTER_SIZE * FILTER_SIZE * sizeof(unsigned char)), NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, imageFromDeviceedge));
    OCL_CHECK(err, err = kernel.setArg(1, imageFromDeviceDilate));
    OCL_CHECK(err, err = kernel.setArg(2, kernelFilterToDevice));
    OCL_CHECK(err, err = kernel.setArg(3, height));
    OCL_CHECK(err, err = kernel.setArg(4, width));

    // Initialize the buffers:
    cl::Event event;
    OCL_CHECK(err, q.enqueueWriteBuffer(kernelFilterToDevice, // buffer on the FPGA
                                            CL_TRUE,              // blocking call
                                            0,                    // buffer offset in bytes
                                            (FILTER_SIZE * FILTER_SIZE * sizeof(unsigned char)), // Size in bytes
                                            structure_element, // Pointer to the data to copy
                                            nullptr, &event));

    // Execute the kernel:
    OCL_CHECK(err, err = q.enqueueTask(kernel, NULL, &event_sp));

    clWaitForEvents(1, (const cl_event*)&event_sp);

    // Profiling
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    // Copy Result from Device Global Memory to Host Local Memory
    q.enqueueReadBuffer(imageFromDeviceDilate, // This buffers data will be read
                            CL_TRUE,         // blocking call
                            0,               // offset
                            (height * width * CH_TYPE),
                            out_img.data, // Data will be stored here
                            nullptr, &event);

    q.finish();

    /////////////////////////////////end of CL call//////////////////////////////////////////////////

    /*				Apply Gaussian mask and call opencv canny function				*/
    cv::Mat img_gray1;
    img_gray1.create(img_gray_1.rows, img_gray_1.cols, img_gray_1.depth());
    AverageGaussian(img_gray_1, img_gray1); // Gaussian filter

#if L1NORM
    cv::Canny(img_gray1, ocv_img, 30.0, 64.0, FILTER_WIDTH, false); // Opencv canny function

#else
    cv::Canny(img_gray1, ocv_img, 30.0, 64.0, FILTER_WIDTH, true); // Opencv canny function
#endif

    absdiff(ocv_img, out_img_edge, diff); // Absolute difference between opencv and hls result
    imwrite("hls.png", out_img_edge);     // Save HLS result
    imwrite("ocv.png", ocv_img);          // Save Opencv result
    imwrite("diff.png", diff);
    // Save difference image
    // Find minimum and maximum differences.
    double minval = 256, maxval = 0;

    int cnt = 0;
    for (int i = 0; i < diff.rows - 0; i++) {
        for (int j = 0; j < diff.cols - 0; j++) {
            uchar v = diff.at<uchar>(i, j);

            if (v > 0) cnt++;
            if (minval > v) minval = v;
            if (maxval < v) maxval = v;
        }
    }

    float err_per = 100.0 * (float)cnt / (diff.rows * diff.cols);

    fprintf(stderr,
            "Minimum error in intensity = %f\n Maximum error in intensity = %f\n Percentage of pixels above error "
            "threshold = %f\nNo of Pixels with Error = %d\n",
            minval, maxval, err_per, cnt);

    fprintf(stderr, "kernel done");
    if (err_per > 2.5f) return 1;
    /*			Destructors			*/
    in_img_1.~Mat();
    img_gray_1.~Mat();
    img_gray1.~Mat();
    hls_img.~Mat();
    ocv_img.~Mat();
    diff.~Mat();

    return 0;
}

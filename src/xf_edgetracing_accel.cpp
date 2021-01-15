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

#include "xf_canny_config.h"

extern "C" {
void edgetracing_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp, ap_uint<OUTPUT_PTR_WIDTH>* img_out, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem4
    #pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem5
    #pragma HLS INTERFACE s_axilite port=img_inp  
    #pragma HLS INTERFACE s_axilite port=img_out
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows     
    #pragma HLS INTERFACE s_axilite port=cols     
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    printf("\nbefore allocate\n");
    xf::cv::Mat<XF_2UC1, HEIGHT, WIDTH, XF_NPPC32> _dst1(rows, cols, img_inp);
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC8> _dst2(rows, cols, img_out);
    printf("\nbefore kernel call\n");
    xf::cv::EdgeTracing<XF_2UC1, XF_8UC1, HEIGHT, WIDTH, XF_NPPC32, XF_NPPC8, XF_USE_URAM>(_dst1, _dst2);
    printf("\nafter kernel call\n");
}
}

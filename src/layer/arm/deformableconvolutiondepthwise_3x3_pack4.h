// 
#pragma once

#include <stdio.h>
#include "benchmark.h"

using namespace ncnn;

#if 1

// void dfmconvdw3x3s1_calc_ofsidx_neon(const Mat& offset, Mat& ofs_idx)
// {

// }

void dfmconvdw3x3s1_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& offset, Mat& ofs_idx2, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int in_w = bottom_blob.w;
    int in_h = bottom_blob.h;
    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const float* bias = _bias;

    const int kernel_size = 9;

#define CALCULATE_OFFSET(k_idx, kh, kw) {\
    float ofs_h = ofsptr[k_idx * 2]; \
    float ofs_w = ofsptr[k_idx * 2 + 1]; \
    int pos_h = oh + (int)round(ofs_h) + kh; \
    int pos_w = ow + (int)round(ofs_w) + kw; \
    if (pos_h < 0) pos_h = 0; \
    if (pos_h >= in_h) pos_h = in_h - 1; \
    if (pos_w < 0) pos_w = 0; \
    if (pos_w >= in_w) pos_w = in_w - 1; \
    ofs_idx_ptr[k_idx] = (pos_h * in_w + pos_w) * 4; \
}

    int *ofs_idx = (int*) ofs_idx2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int oh = 0; oh < outh; oh++)
    {
        float* ofsptr = (float*)offset + oh * outw * kernel_size * 2;
        int* ofs_idx_ptr = (int*)ofs_idx + oh * outw * kernel_size;

        for (int ow = 0; ow < outw; ow++)
        {
            CALCULATE_OFFSET(0, 0, 0);
            CALCULATE_OFFSET(1, 0, 1);
            CALCULATE_OFFSET(2, 0, 2);

            CALCULATE_OFFSET(3, 1, 0);
            CALCULATE_OFFSET(4, 1, 1);
            CALCULATE_OFFSET(5, 1, 2);

            CALCULATE_OFFSET(6, 2, 0);
            CALCULATE_OFFSET(7, 2, 1);
            CALCULATE_OFFSET(8, 2, 2);

            ofsptr += kernel_size * 2;
            ofs_idx_ptr += kernel_size;
        }
    }

#undef CALCULATE_OFFSET

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g=0; g<group; g++)
    {
        Mat out = top_blob.channel(g);
        const Mat img0 = bottom_blob.channel(g);
        const float* k0 = kernel.row(g);

        float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + g * 4) : vdupq_n_f32(0.f);

        const float* input = (const float*) img0;

        float32x4_t _k00 = vld1q_f32(k0);
        float32x4_t _k01 = vld1q_f32(k0+4);
        float32x4_t _k02 = vld1q_f32(k0+8);
        float32x4_t _k10 = vld1q_f32(k0+12);
        float32x4_t _k11 = vld1q_f32(k0+16);
        float32x4_t _k12 = vld1q_f32(k0+20);
        float32x4_t _k20 = vld1q_f32(k0+24);
        float32x4_t _k21 = vld1q_f32(k0+28);
        float32x4_t _k22 = vld1q_f32(k0+32);

        int* ofs_idx_ptr = (int*)ofs_idx;
        for (int oh = 0; oh < outh; oh++)
        {
            float* output = out.row(oh);
            for (int ow = 0; ow < outw; ow++)
            {
                float32x4_t _dst = _bias0;

                int idx00 = ofs_idx_ptr[0];
                int idx01 = ofs_idx_ptr[1];
                int idx02 = ofs_idx_ptr[2];
                int idx10 = ofs_idx_ptr[3];
                int idx11 = ofs_idx_ptr[4];
                int idx12 = ofs_idx_ptr[5];
                int idx20 = ofs_idx_ptr[6];
                int idx21 = ofs_idx_ptr[7];
                int idx22 = ofs_idx_ptr[8];

                float32x4_t _input00 = vld1q_f32(input + idx00);
                float32x4_t _input01 = vld1q_f32(input + idx01);
                float32x4_t _input02 = vld1q_f32(input + idx02);
                float32x4_t _input10 = vld1q_f32(input + idx10);
                float32x4_t _input11 = vld1q_f32(input + idx11);
                float32x4_t _input12 = vld1q_f32(input + idx12);
                float32x4_t _input20 = vld1q_f32(input + idx20);
                float32x4_t _input21 = vld1q_f32(input + idx21);
                float32x4_t _input22 = vld1q_f32(input + idx22);

                _dst = vmlaq_f32(_dst, _input00, _k00);
                _dst = vmlaq_f32(_dst, _input01, _k01);
                _dst = vmlaq_f32(_dst, _input02, _k02);
                _dst = vmlaq_f32(_dst, _input10, _k10);
                _dst = vmlaq_f32(_dst, _input11, _k11);
                _dst = vmlaq_f32(_dst, _input12, _k12);
                _dst = vmlaq_f32(_dst, _input20, _k20);
                _dst = vmlaq_f32(_dst, _input21, _k21);
                _dst = vmlaq_f32(_dst, _input22, _k22);

                vst1q_f32(output + ow * 4, _dst);

                ofs_idx_ptr += kernel_size;
            }
        }
    }
}

void dfmconvdw3x3s2_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& offset, Mat& ofs_idx2, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int in_w = bottom_blob.w;
    int in_h = bottom_blob.h;
    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const float* bias = _bias;

    const int kernel_size = 9;

#define CALCULATE_OFFSET(k_idx, kh, kw) {\
    float ofs_h = ofsptr[k_idx * 2]; \
    float ofs_w = ofsptr[k_idx * 2 + 1]; \
    int pos_h = oh * 2 + (int)round(ofs_h) + kh; \
    int pos_w = ow * 2 + (int)round(ofs_w) + kw; \
    if (pos_h < 0) pos_h = 0; \
    if (pos_h >= in_h) pos_h = in_h - 1; \
    if (pos_w < 0) pos_w = 0; \
    if (pos_w >= in_w) pos_w = in_w - 1; \
    ofs_idx_ptr[k_idx] = (pos_h * in_w + pos_w) * 4; \
}

    int *ofs_idx = (int*) ofs_idx2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int oh = 0; oh < outh; oh++)
    {
        float* ofsptr = (float*)offset + oh * outw * kernel_size * 2;
        int* ofs_idx_ptr = (int*)ofs_idx + oh * outw * kernel_size;

        for (int ow = 0; ow < outw; ow++)
        {
            CALCULATE_OFFSET(0, 0, 0);
            CALCULATE_OFFSET(1, 0, 1);
            CALCULATE_OFFSET(2, 0, 2);

            CALCULATE_OFFSET(3, 1, 0);
            CALCULATE_OFFSET(4, 1, 1);
            CALCULATE_OFFSET(5, 1, 2);

            CALCULATE_OFFSET(6, 2, 0);
            CALCULATE_OFFSET(7, 2, 1);
            CALCULATE_OFFSET(8, 2, 2);

            ofsptr += kernel_size * 2;
            ofs_idx_ptr += kernel_size;
        }
    }

#undef CALCULATE_OFFSET

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g=0; g<group; g++)
    {
        Mat out = top_blob.channel(g);
        const Mat img0 = bottom_blob.channel(g);
        const float* k0 = kernel.row(g);

        float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + g * 4) : vdupq_n_f32(0.f);

        const float* input = (const float*) img0;
        //float* output = (float*)out;

        float32x4_t _k00 = vld1q_f32(k0);
        float32x4_t _k01 = vld1q_f32(k0+4);
        float32x4_t _k02 = vld1q_f32(k0+8);
        float32x4_t _k10 = vld1q_f32(k0+12);
        float32x4_t _k11 = vld1q_f32(k0+16);
        float32x4_t _k12 = vld1q_f32(k0+20);
        float32x4_t _k20 = vld1q_f32(k0+24);
        float32x4_t _k21 = vld1q_f32(k0+28);
        float32x4_t _k22 = vld1q_f32(k0+32);

        int* ofs_idx_ptr = (int*)ofs_idx;
        for (int oh = 0; oh < outh; oh++)
        {
            float* output = out.row(oh);
            for (int ow = 0; ow < outw; ow++)
            {
                float32x4_t _dst = _bias0;

                int idx00 = ofs_idx_ptr[0];
                int idx01 = ofs_idx_ptr[1];
                int idx02 = ofs_idx_ptr[2];
                int idx10 = ofs_idx_ptr[3];
                int idx11 = ofs_idx_ptr[4];
                int idx12 = ofs_idx_ptr[5];
                int idx20 = ofs_idx_ptr[6];
                int idx21 = ofs_idx_ptr[7];
                int idx22 = ofs_idx_ptr[8];

                float32x4_t _input00 = vld1q_f32(input + idx00);
                float32x4_t _input01 = vld1q_f32(input + idx01);
                float32x4_t _input02 = vld1q_f32(input + idx02);
                float32x4_t _input10 = vld1q_f32(input + idx10);
                float32x4_t _input11 = vld1q_f32(input + idx11);
                float32x4_t _input12 = vld1q_f32(input + idx12);
                float32x4_t _input20 = vld1q_f32(input + idx20);
                float32x4_t _input21 = vld1q_f32(input + idx21);
                float32x4_t _input22 = vld1q_f32(input + idx22);

                _dst = vmlaq_f32(_dst, _input00, _k00);
                _dst = vmlaq_f32(_dst, _input01, _k01);
                _dst = vmlaq_f32(_dst, _input02, _k02);
                _dst = vmlaq_f32(_dst, _input10, _k10);
                _dst = vmlaq_f32(_dst, _input11, _k11);
                _dst = vmlaq_f32(_dst, _input12, _k12);
                _dst = vmlaq_f32(_dst, _input20, _k20);
                _dst = vmlaq_f32(_dst, _input21, _k21);
                _dst = vmlaq_f32(_dst, _input22, _k22);

                vst1q_f32(output + ow * 4, _dst);

                ofs_idx_ptr += kernel_size;
            }
        }
    }
}

#else
static void dfmconvdw3x3s1_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, Mat& offset, Mat& ofs_idx2, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int in_w = bottom_blob.w;
    int in_h = bottom_blob.h;
    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const float* bias = _bias;

    const int kernel_size = 9;

#define CALCULATE_OFFSET(k_idx, kh, kw) {\
    float ofs_h = ofsptr[k_idx * 2]; \
    float ofs_w = ofsptr[k_idx * 2 + 1]; \
    int pos_h = oh + (int)round(ofs_h) + kh; \
    int pos_w = ow + (int)round(ofs_w) + kw; \
    if (pos_h < 0) pos_h = 0; \
    if (pos_h >= in_h) pos_h = in_h - 1; \
    if (pos_w < 0) pos_w = 0; \
    if (pos_w >= in_w) pos_w = in_w - 1; \
    ofs_idx_ptr[k_idx] = (pos_h * in_w + pos_w) * 4; \
}

    //printf("ofs_idx_ptr[%p]: %d\n", ofs_idx_ptr, ofs_idx_ptr[k_idx]); \
    //printf("oh=%d, ow=%d, idx=%d, k_h=%d, k_w=%d, in_h=%d, in_w=%d, pos_h=%d, pos_w=%d\n", oh, ow, k_idx, kh, kw, in_h, in_w, pos_h, pos_w);\
    //printf("ofsptr[%p]: %f, %f\n", ofsptr, ofsptr[2 * k_idx], ofsptr[2 * k_idx + 1]); \
    //printf("ofs_idx_ptr[%p]: %d\n", ofs_idx_ptr, ofs_idx_ptr[k_idx]); \

// printf("    dfmconvdw3x3s1_pack4_neon gogogo\n\n");

    int *ofs_idx = (int*) ofs_idx2;

    // double start = ncnn::get_current_time();

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int oh = 0; oh < outh; oh++)
    {
        float* ofsptr = (float*)offset + oh * outw * kernel_size * 2;
        int* ofs_idx_ptr = (int*)ofs_idx + oh * outw * kernel_size;

        for (int ow = 0; ow < outw; ow++)
        {
            CALCULATE_OFFSET(0, 0, 0);
            CALCULATE_OFFSET(1, 0, 1);
            CALCULATE_OFFSET(2, 0, 2);

            CALCULATE_OFFSET(3, 1, 0);
            CALCULATE_OFFSET(4, 1, 1);
            CALCULATE_OFFSET(5, 1, 2);

            CALCULATE_OFFSET(6, 2, 0);
            CALCULATE_OFFSET(7, 2, 1);
            CALCULATE_OFFSET(8, 2, 2);

            ofsptr += kernel_size * 2;
            ofs_idx_ptr += kernel_size;
        }
    }

#undef CALCULATE_OFFSET

    // double end = ncnn::get_current_time();
    // double time = end - start;

    // printf("calculate offset: %f ms\n", time);

// printf("    dfmconvdw3x3s1_pack4_neon calc-ed pos\n\n");


    // start = ncnn::get_current_time();

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g=0; g<group; g++)
    {
// printf("    dfmconvdw3x3s1_pack4_neon in\n\n");
        Mat out = top_blob.channel(g);
        const Mat img0 = bottom_blob.channel(g);
        const float* k0 = kernel.row(g);

// printf("    dfmconvdw3x3s1_pack4_neon bias\n\n");
        float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + g * 4) : vdupq_n_f32(0.f);

        // float* outptr0 = out.row(0);
        // float* outptr1 = out.row(1);

        // const float* r0 = img0.row(0);
        // const float* r1 = img0.row(1);
        // const float* r2 = img0.row(2);
        // const float* r3 = img0.row(3);

        const float* input = (const float*) img0;
        float* output = (float*)out;

        float32x4_t _k00 = vld1q_f32(k0);
        float32x4_t _k01 = vld1q_f32(k0+4);
        float32x4_t _k02 = vld1q_f32(k0+8);
        float32x4_t _k10 = vld1q_f32(k0+12);
        float32x4_t _k11 = vld1q_f32(k0+16);
        float32x4_t _k12 = vld1q_f32(k0+20);
        float32x4_t _k20 = vld1q_f32(k0+24);
        float32x4_t _k21 = vld1q_f32(k0+28);
        float32x4_t _k22 = vld1q_f32(k0+32);


// printf("    dfmconvdw3x3s1_pack4_neon heavy calc\n\n");
        int* ofs_idx_ptr = (int*)ofs_idx2;
        for (int oh = 0; oh < outh; oh++)
        {
// printf("    int oh = 0; %d < %d; oh++\n\n", oh, outh);
            for (int ow = 0; ow < outw; ow++)
            {
// printf("    int ow = 0; %d < %d; ow++\n\n", ow, outw);
                int idx00 = ofs_idx_ptr[0];
                int idx01 = ofs_idx_ptr[1];
                int idx02 = ofs_idx_ptr[2];
                int idx10 = ofs_idx_ptr[3];
                int idx11 = ofs_idx_ptr[4];
                int idx12 = ofs_idx_ptr[5];
                int idx20 = ofs_idx_ptr[6];
                int idx21 = ofs_idx_ptr[7];
                int idx22 = ofs_idx_ptr[8];

// printf("    float32x4_t _dst = _bias0;\n\n", ow, outw);
                float32x4_t _dst = _bias0;

printf("    int idx00 = %d; %d (%p)\n", idx00, ofs_idx_ptr[0], ofs_idx_ptr);
printf("    int idx01 = %d; %d (%p)\n", idx01, ofs_idx_ptr[1], ofs_idx_ptr);
printf("    int idx02 = %d; %d (%p)\n", idx02, ofs_idx_ptr[2], ofs_idx_ptr);
printf("    int idx10 = %d; %d (%p)\n", idx10, ofs_idx_ptr[3], ofs_idx_ptr);
printf("    int idx11 = %d; %d (%p)\n", idx11, ofs_idx_ptr[4], ofs_idx_ptr);
printf("    int idx12 = %d; %d (%p)\n", idx12, ofs_idx_ptr[5], ofs_idx_ptr);
printf("    int idx20 = %d; %d (%p)\n", idx20, ofs_idx_ptr[6], ofs_idx_ptr);
printf("    int idx21 = %d; %d (%p)\n", idx21, ofs_idx_ptr[7], ofs_idx_ptr);
printf("    int idx22 = %d; %d (%p)\n", idx22, ofs_idx_ptr[8], ofs_idx_ptr);

                float32x4_t _input00 = vld1q_f32(input + idx00);
                float32x4_t _input01 = vld1q_f32(input + idx01);
                float32x4_t _input02 = vld1q_f32(input + idx02);
                float32x4_t _input10 = vld1q_f32(input + idx10);
                float32x4_t _input11 = vld1q_f32(input + idx11);
                float32x4_t _input12 = vld1q_f32(input + idx12);
                float32x4_t _input20 = vld1q_f32(input + idx20);
                float32x4_t _input21 = vld1q_f32(input + idx21);
                float32x4_t _input22 = vld1q_f32(input + idx22);

// printf("    _dst = vmlaq_f32(_dst, _input00, _k00)\n\n");

                _dst = vmlaq_f32(_dst, _input00, _k00);
                _dst = vmlaq_f32(_dst, _input01, _k01);
                _dst = vmlaq_f32(_dst, _input02, _k02);
                _dst = vmlaq_f32(_dst, _input10, _k10);
                _dst = vmlaq_f32(_dst, _input11, _k11);
                _dst = vmlaq_f32(_dst, _input12, _k12);
                _dst = vmlaq_f32(_dst, _input20, _k20);
                _dst = vmlaq_f32(_dst, _input21, _k21);
                _dst = vmlaq_f32(_dst, _input22, _k22);

// printf("    dfmconvdw3x3s1_pack4_neon set\n\n");
                vst1q_f32(output + (oh * outw + ow) * 4, _dst);

// printf("    ofs_idx_ptr += kernel_size\n\n");
                ofs_idx_ptr += kernel_size;
            }
        }
    }

    // end = ncnn::get_current_time();
    // time = end - start;
    // printf("calculate conv: %f ms\n", time);

// printf("    dfmconvdw3x3s1_pack4_neon finish\n\n");
}
#endif


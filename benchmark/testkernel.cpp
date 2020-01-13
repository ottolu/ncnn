#include <float.h>
#include <stdio.h>
#include <string.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#ifdef _WIN32
#include <algorithm>
#include <windows.h> // Sleep()
#else
#include <unistd.h> // sleep()
#endif

#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"
#include "net.h"

using namespace ncnn;
using namespace std;

#include "../src/layer/arm/deformableconvolutiondepthwise_3x3_pack4.h"

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

std::string string_trim(std::string input)
{
    int begin = 0, end = input.size() - 1;
    for (; begin < input.size(); begin++)
    {
        if (input[begin] != ' ') break;
    }
    if (begin == input.size()) return "";
    for (; end >= 0; end--)
    {
        if (input[end] != ' ') break;
    }
    return input.substr(begin, end - begin + 1);
}

std::vector<std::string> string_split(std::string input, char ch)
{
    std::vector<std::string> ret;

    size_t end = input.find(ch);
    size_t begin = 0;

    while (end != std::string::npos)
    {
        ret.push_back(input.substr(begin, end - begin));
 
        begin = end + 1;
        end = input.find(ch, begin);
    }
    ret.push_back(input.substr(begin));
    return ret;
}

#if 0

int main(int argc, char** argv)
{
    int loop_count = 5;
    int num_threads = ncnn::get_cpu_count();
    int powersave = 0;
    int gpu_device = -1;

    string input_file_name;

    if (argc < 5)
    {
        fprintf(stderr, "error input\n");
    }

    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);

    Option opt;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.lightmode = true;
    opt.num_threads = num_threads;
    opt.use_packing_layout = true;

    ncnn::set_cpu_powersave(powersave);
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    // fprintf(stderr, "loop_count = %d\n", loop_count);
    // fprintf(stderr, "num_threads = %d\n", num_threads);
    // fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    // fprintf(stderr, "gpu_device = %d\n", gpu_device);

    int inw, inh, inc, kernel_h, kernel_w, num_output, outw, outh, elempack, stride_h, stride_w;

    elempack = 4;
    size_t elemsize = 4;

    inh = inw = 5;
    inc = 8;
    kernel_h = kernel_w = 3;
    num_output = 8;
    stride_h = stride_w = 1;

    num_output = atoi(argv[1]);
    inw = atoi(argv[2]);
    inh = atoi(argv[3]);
    inc = atoi(argv[4]);
    
    outw = (inw - kernel_w) / stride_w + 1;
    outh = (inh - kernel_h) / stride_h + 1;

    double start = ncnn::get_current_time();

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

    Mat bottom_blob, top_blob, kernel, offset, ofs_idx, bias;
    bottom_blob.create(inw, inh, inc / elempack, elemsize, elempack, opt.blob_allocator);
    top_blob.create(outw, outh, num_output / elempack, elemsize, elempack, opt.blob_allocator);
    kernel.create(kernel_w, kernel_h, num_output / elempack, elemsize, elempack, opt.blob_allocator);
    bias.create(1, 1, num_output / elempack, elemsize, elempack, opt.blob_allocator);

    offset.create(2 * kernel_h * kernel_w * outh * outw, 1, elemsize, 1);
    ofs_idx.create(kernel_h * kernel_w * outw * outh, 1, elemsize, 1);

    double end = ncnn::get_current_time();
    double create_blob_time = end - start;

    float offset_range = 10.0;
    float* offset_ptr = (float*)offset;
    for (int i = 0; i < 2 * kernel_h * kernel_w * outh * outw; ++i) {
        offset_ptr[i] = rand() * 1.0f / RAND_MAX * offset_range * 2.0 - offset_range;
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;
    for (int loop_index = 0; loop_index < loop_count; loop_index++)
    {
        start = ncnn::get_current_time();
        dfmconvdw3x3s1_pack4_neon(bottom_blob, top_blob, kernel, offset, ofs_idx, bias, opt);
        end = ncnn::get_current_time();
        double time = end - start;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
    }
    time_avg /= loop_count;

    fprintf(stderr, "%s min = %.2f  max = %.2f  avg = %.2f\n", line.c_str(), time_min, time_max, time_avg);
    printf("%lf\n", time_avg);

    // sleep(10);

    return 0;
}

#else
int main(int argc, char** argv)
{
    int loop_count = 100;
    int num_threads = ncnn::get_cpu_count();
    int powersave = 0;
    int gpu_device = -1;
    int ofs_range = 10;

    string input_file_name;

    if (argc < 2)
    {
        fprintf(stderr, "missing input txt file\n");
    }
    if (argc >= 2)
    {
        input_file_name = argv[1];
    }
    if (argc >= 3)
    {
        loop_count = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        num_threads = atoi(argv[3]);
    }
    if (argc >= 5)
    {
        ofs_range = atoi(argv[4]);
    }

    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);

    Option opt;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.lightmode = false;
    opt.num_threads = num_threads;
    opt.use_packing_layout = true;

    ncnn::set_cpu_powersave(powersave);
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    fprintf(stderr, "loop_count = %d\n", loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);
    fprintf(stderr, "ofs_range = %d\n", ofs_range);

    char tmpbuffer[256];
    fstream file;
    file.open(input_file_name, std::ios::in);

    double total_time = 0.0;

    while (!file.eof())
    {
        file.getline (tmpbuffer,256);
        string line = string_trim(tmpbuffer);

        if (line == "") continue;
        if (line[0] == '#') continue; // skip comment

        auto words = string_split(line, ' ');

        if (words.size() != 4) 
        {
            fprintf(stderr, "\nwrong format: %s\n\n", line.c_str());
            continue;
        }

        int inw, inh, inc, kernel_h, kernel_w, num_output, outw, outh, elempack, stride_h, stride_w;
    
        elempack = 4;
        size_t elemsize = 4;

        inh = inw = 5;
        inc = 8;
        kernel_h = kernel_w = 3;
        num_output = 8;
        stride_h = stride_w = 1;

        num_output = atoi(words[0].c_str());
        inw = atoi(words[1].c_str());
        inh = atoi(words[2].c_str());
        inc = atoi(words[3].c_str());
        
        outw = (inw - kernel_w) / stride_w + 1;
        outh = (inh - kernel_h) / stride_h + 1;

        // fprintf(stderr, "Create blobs\n");

        double start = ncnn::get_current_time();

        Mat bottom_blob, top_blob, kernel, offset, ofs_idx, bias;
        bottom_blob.create(inw, inh, inc / elempack, elemsize, elempack);
        top_blob.create(outw, outh, num_output / elempack, elemsize, elempack);
        kernel.create(kernel_w, kernel_h, num_output / elempack, elemsize, elempack);
        bias.create(1, 1, num_output / elempack, elemsize, elempack);

        offset.create(2 * kernel_h * kernel_w * outh * outw, 1, elemsize, 1);
        ofs_idx.create(kernel_h * kernel_w * outw * outh, 1, elemsize, 1);

        double end = ncnn::get_current_time();
        double create_blob_time = end - start;

        float offset_range = (float)ofs_range;
        float* offset_ptr = (float*)offset;
        for (int i = 0; i < 2 * kernel_h * kernel_w * outh * outw; ++i) {
            offset_ptr[i] = rand() * 1.0f / RAND_MAX * offset_range * 2.0 - offset_range;
        }

        double time_min = DBL_MAX;
        double time_max = -DBL_MAX;
        double time_avg = 0;
        // warmup
        for (int loop_index = 0; loop_index < loop_count; loop_index++)
        {
            dfmconvdw3x3s1_pack4_neon(bottom_blob, top_blob, kernel, offset, ofs_idx, bias, opt);
        }
        start = ncnn::get_current_time();
        for (int loop_index = 0; loop_index < loop_count; loop_index++)
        {
            dfmconvdw3x3s1_pack4_neon(bottom_blob, top_blob, kernel, offset, ofs_idx, bias, opt);
        }
        end = ncnn::get_current_time();
        double time = end - start;
        time /= loop_count;

        fprintf(stderr, "%s: %lf\n", line.c_str(), time);

        total_time += time;
        // fprintf(stderr, "file.eof(): %d\n", file.eof());
    }

    fprintf(stderr, "total time: %lf\n", total_time);


#if 0
    int inw, inh, inc, kernel_h, kernel_w, num_output, outw, outh, elempack, stride_h, stride_w;
    
    elempack = 4;
    size_t elemsize = 4;

    inh = inw = 5;
    inc = 8;
    kernel_h = kernel_w = 3;
    num_output = 8;
    stride_h = stride_w = 1;

    outw = (inw - kernel_w) / stride_w + 1;
    outh = (inh - kernel_h) / stride_h + 1;

    fprintf(stderr, "Create blobs\n");

    double start = ncnn::get_current_time();

    bottom_blob.create(inw, inh, inc / elempack, elemsize, elempack, opt.blob_allocator);
    top_blob.create(outw, outh, num_output / elempack, elemsize, elempack, opt.blob_allocator);
    kernel.create(kernel_w, kernel_h, num_output / elempack, elemsize, elempack, opt.blob_allocator);
    bias.create(1, 1, num_output / elempack, elemsize, elempack, opt.blob_allocator);

    offset.create(2 * kernel_h * kernel_w * outh * outw, 1, elemsize, 1);
    ofs_idx.create(kernel_h * kernel_w * outw * outh, 1, elemsize, 1);

    double end = ncnn::get_current_time();
    double time = end - start;

    fprintf(stderr, "call dfmconvdw3x3s1_pack4_neon\n");

    dfmconvdw3x3s1_pack4_neon(bottom_blob, top_blob, kernel, offset, ofs_idx, bias, opt);
    
    fprintf(stderr, "finish dfmconvdw3x3s1_pack4_neon\n");
#endif

    return 0;
}

#endif
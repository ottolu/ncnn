#include <float.h>
#include <stdio.h>
#include <string.h>

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

#include "../src/layer/arm/deformableconvolutiondepthwise_3x3_pack4.h"

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

int main(int argc, char** argv)
{
    int loop_count = 1;
    int num_threads = 1;
    int powersave = 0;
    int gpu_device = -1;

    if (argc >= 2)
    {
        loop_count = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        num_threads = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        powersave = atoi(argv[3]);
    }
    if (argc >= 5)
    {
        gpu_device = atoi(argv[4]);
    }

    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);

    Mat bottom_blob, top_blob, kernel, offset, ofs_idx, bias;
    Option opt;

    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.lightmode = true;
    opt.num_threads = num_threads;
    opt.use_packing_layout = true;

    ncnn::set_cpu_powersave(powersave);
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    fprintf(stderr, "loop_count = %d\n", loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);

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

    return 0;
}
#ifndef LAYER_DEFORMABLECONVOLUTIONDEPTHWISE_H
#define LAYER_DEFORMABLECONVOLUTIONDEPTHWISE_H

#include "layer.h"

namespace ncnn {

class DeformableConvolutionDepthWise : public Layer
{
public:
    DeformableConvolutionDepthWise();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    //virtual int create_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    //int forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    // param
    int num_output;
    int kernel_w;
    int kernel_h;
    //int dilation_w;
    //int dilation_h;
    int stride_w;
    int stride_h;
    int pad_left;// -233=SAME_UPPER -234=SAME_LOWER
    int pad_right;
    int pad_top;
    int pad_bottom;
    float pad_value;
    int bias_term;

    int weight_data_size;
    int offset_data_size;
    int group;

    int int8_scale_term;

    // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    Mat activation_params;

    // model
    Mat weight_data;
    Mat bias_data;
    Mat offset_data;
    Mat tmp_data;

    Mat weight_data_int8_scales;
    Mat bottom_blob_int8_scales;
    float top_blob_int8_scale;

    bool use_int8_requantize;
};

} // namespace ncnn

#endif // LAYER_DEFORMABLECONVOLUTIONDEPTHWISE_H
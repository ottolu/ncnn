#ifndef LAYER_DEFORMABLECONVOLUTIONDEPTHWISE_ARM_H
#define LAYER_DEFORMABLECONVOLUTIONDEPTHWISE_ARM_H

#include "deformableconvolutiondepthwise.h"

namespace ncnn {

class DeformableConvolutionDepthWise_arm : virtual public DeformableConvolutionDepthWise
{
public:
    DeformableConvolutionDepthWise_arm();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    //int forward_int8_arm(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    Layer* activation;
    std::vector<ncnn::Layer*> group_ops;

    // buffer

    // packing
    Mat weight_data_pack4;
    Mat tmp_data;
    Mat offset_data1;
};

} // namespace ncnn

#endif // LAYER_DEFORMABLECONVOLUTIONDEPTHWISE_ARM_H
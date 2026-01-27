#include "myproject.h"
#include <sycl/ext/intel/experimental/task_sequence.hpp>

// hls-fpga-machine-learning insert weights
#include "weights/w3.h"
#include "weights/b3.h"
#include "weights/w6.h"
#include "weights/b6.h"
#include "weights/w9.h"
#include "weights/b9.h"
#include "weights/w12.h"
#include "weights/b12.h"


#include "parameters.h"


// The inter-task pipes need to be declared in the global scope
// hls-fpga-machine-learning insert inter-task pipes

using sycl::ext::intel::experimental::task_sequence;

void Myproject::operator()() const {
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning read in
    auto input_layer = InputLayerPipe::read();

    // hls-fpga-machine-learning declare task sequences

    // hls-fpga-machine-learning insert layers

    [[intel::fpga_register]] q_dense_t layer3_out;
    nnet::dense_resource<input_layer_t, q_dense_t, config3>(input_layer, layer3_out);
    [[intel::fpga_register]] q_dense_relu_t layer4_out;
    nnet::relu<q_dense_t, q_dense_relu_t, relu_config4>(layer3_out, layer4_out);
    [[intel::fpga_register]] q_dense_1_iq_t layer5_out;
    nnet::q_dense_1_iq<q_dense_relu_t, q_dense_1_iq_t>(layer4_out, layer5_out);
    [[intel::fpga_register]] q_dense_1_t layer6_out;
    nnet::dense_resource<q_dense_1_iq_t, q_dense_1_t, config6>(layer5_out, layer6_out);
    [[intel::fpga_register]] q_dense_1_relu_t layer7_out;
    nnet::relu<q_dense_1_t, q_dense_1_relu_t, relu_config7>(layer6_out, layer7_out);
    [[intel::fpga_register]] q_dense_2_iq_t layer8_out;
    nnet::q_dense_2_iq<q_dense_1_relu_t, q_dense_2_iq_t>(layer7_out, layer8_out);
    [[intel::fpga_register]] q_dense_2_t layer9_out;
    nnet::dense_resource<q_dense_2_iq_t, q_dense_2_t, config9>(layer8_out, layer9_out);
    [[intel::fpga_register]] q_dense_2_relu_t layer10_out;
    nnet::relu<q_dense_2_t, q_dense_2_relu_t, relu_config10>(layer9_out, layer10_out);
    [[intel::fpga_register]] q_dense_3_iq_t layer11_out;
    nnet::q_dense_3_iq<q_dense_2_relu_t, q_dense_3_iq_t>(layer10_out, layer11_out);
    [[intel::fpga_register]] result_t layer12_out;
    nnet::dense_resource<q_dense_3_iq_t, result_t, config12>(layer11_out, layer12_out);

    // hls-fpga-machine-learning return
    Layer12OutPipe::write(layer12_out);
}

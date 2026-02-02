#include "myproject.h"
#include <sycl/ext/intel/experimental/task_sequence.hpp>

// hls-fpga-machine-learning insert weights
#include "weights/w55.h"
#include "weights/b55.h"
#include "weights/w56.h"
#include "weights/b56.h"
#include "weights/w9.h"
#include "weights/b9.h"
#include "weights/w11.h"
#include "weights/b11.h"
#include "weights/w13.h"
#include "weights/b13.h"
#include "weights/w16.h"
#include "weights/b16.h"
#include "weights/w19.h"
#include "weights/b19.h"
#include "weights/w27.h"
#include "weights/b27.h"
#include "weights/w57.h"
#include "weights/b57.h"
#include "weights/w58.h"
#include "weights/b58.h"
#include "weights/w59.h"
#include "weights/b59.h"
#include "weights/w60.h"
#include "weights/b60.h"
#include "weights/w51.h"
#include "weights/b51.h"
#include "weights/w54.h"
#include "weights/b54.h"


#include "parameters.h"


// The inter-task pipes need to be declared in the global scope
// hls-fpga-machine-learning insert inter-task pipes

using sycl::ext::intel::experimental::task_sequence;

void Myproject::operator()() const {
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning read in
    auto input_layer_7 = InputLayer7Pipe::read();

    // hls-fpga-machine-learning declare task sequences

    // hls-fpga-machine-learning insert layers

    [[intel::fpga_register]] q_dense_56_iq_t layer2_out;
    nnet::q_dense_56_iq<input_layer_7_t, q_dense_56_iq_t>(input_layer_7, layer2_out);
    [[intel::fpga_register]] q_dense_56_t layer55_out;
    nnet::pointwise_conv_1d_cl<q_dense_56_iq_t, q_dense_56_t, config61>(layer2_out, layer55_out, w55, b55);
    [[intel::fpga_register]] q_dense_56_relu_t layer4_out;
    nnet::relu<q_dense_56_t, q_dense_56_relu_t, relu_config4>(layer55_out, layer4_out);
    [[intel::fpga_register]] q_dense_57_iq_t layer5_out;
    nnet::q_dense_57_iq<q_dense_56_relu_t, q_dense_57_iq_t>(layer4_out, layer5_out);
    [[intel::fpga_register]] q_dense_57_t layer56_out;
    nnet::pointwise_conv_1d_cl<q_dense_57_iq_t, q_dense_57_t, config62>(layer5_out, layer56_out, w56, b56);
    [[intel::fpga_register]] q_dense_57_relu_t layer7_out;
    nnet::relu<q_dense_57_t, q_dense_57_relu_t, relu_config7>(layer56_out, layer7_out);
    [[intel::fpga_register]] q_einsum_dense_iq_t layer8_out;
    nnet::q_einsum_dense_iq<q_dense_57_relu_t, q_einsum_dense_iq_t>(layer7_out, layer8_out);
    [[intel::fpga_register]] q_einsum_dense_t layer9_out;
    nnet::einsum_dense<q_einsum_dense_iq_t, q_einsum_dense_t, config9>(layer8_out, layer9_out, w9, b9);
    [[intel::fpga_register]] q_einsum_dense_1_iq_t layer10_out;
    nnet::q_einsum_dense_1_iq<q_dense_57_relu_t, q_einsum_dense_1_iq_t>(layer7_out, layer10_out);
    [[intel::fpga_register]] q_einsum_dense_1_t layer11_out;
    nnet::einsum_dense<q_einsum_dense_1_iq_t, q_einsum_dense_1_t, config11>(layer10_out, layer11_out, w11, b11);
    [[intel::fpga_register]] mha1_query_iq_t layer12_out;
    nnet::mha1_query_iq<q_dense_57_relu_t, mha1_query_iq_t>(layer7_out, layer12_out);
    [[intel::fpga_register]] mha1_query_t layer13_out;
    nnet::einsum_dense<mha1_query_iq_t, mha1_query_t, config13>(layer12_out, layer13_out, w13, b13);
    [[intel::fpga_register]] mha1_query_oq_t layer14_out;
    nnet::mha1_query_oq<mha1_query_t, mha1_query_oq_t>(layer13_out, layer14_out);
    [[intel::fpga_register]] mha1_key_iq_t layer15_out;
    nnet::mha1_key_iq<q_einsum_dense_t, mha1_key_iq_t>(layer9_out, layer15_out);
    [[intel::fpga_register]] mha1_key_t layer16_out;
    nnet::einsum_dense<mha1_key_iq_t, mha1_key_t, config16>(layer15_out, layer16_out, w16, b16);
    [[intel::fpga_register]] mha1_key_oq_t layer17_out;
    nnet::mha1_key_oq<mha1_key_t, mha1_key_oq_t>(layer16_out, layer17_out);
    [[intel::fpga_register]] mha1_value_iq_t layer18_out;
    nnet::mha1_value_iq<q_einsum_dense_1_t, mha1_value_iq_t>(layer11_out, layer18_out);
    [[intel::fpga_register]] mha1_value_t layer19_out;
    nnet::einsum_dense<mha1_value_iq_t, mha1_value_t, config19>(layer18_out, layer19_out, w19, b19);
    [[intel::fpga_register]] mha1_value_oq_t layer20_out;
    nnet::mha1_value_oq<mha1_value_t, mha1_value_oq_t>(layer19_out, layer20_out);
    [[intel::fpga_register]] mha1_mha1_QK_t layer21_out;
    nnet::einsum<mha1_key_oq_t, mha1_query_oq_t, mha1_mha1_QK_t, config21>(layer17_out, layer14_out, layer21_out);
    [[intel::fpga_register]] mha1_q_softmax_iq_t layer22_out;
    nnet::mha1_q_softmax_iq<mha1_mha1_QK_t, mha1_q_softmax_iq_t>(layer21_out, layer22_out);
    [[intel::fpga_register]] mha1_q_softmax_t layer23_out;
    nnet::softmax_multidim<mha1_q_softmax_iq_t, mha1_q_softmax_t, softmax_config23>(layer22_out, layer23_out);
    [[intel::fpga_register]] mha1_q_softmax_oq_t layer24_out;
    nnet::mha1_q_softmax_oq<mha1_q_softmax_t, mha1_q_softmax_oq_t>(layer23_out, layer24_out);
    [[intel::fpga_register]] mha1_mha1_aV_t layer25_out;
    nnet::einsum<mha1_q_softmax_oq_t, mha1_value_oq_t, mha1_mha1_aV_t, config25>(layer24_out, layer20_out, layer25_out);
    [[intel::fpga_register]] mha1_attention_output_iq_t layer26_out;
    nnet::mha1_attention_output_iq<mha1_mha1_aV_t, mha1_attention_output_iq_t>(layer25_out, layer26_out);
    [[intel::fpga_register]] mha1_attention_output_t layer27_out;
    nnet::einsum_dense<mha1_attention_output_iq_t, mha1_attention_output_t, config27>(layer26_out, layer27_out, w27, b27);
    [[intel::fpga_register]] quantizer_t layer28_out;
    nnet::quantizer<mha1_attention_output_t, quantizer_t>(layer27_out, layer28_out);
    [[intel::fpga_register]] quantizer_1_t layer29_out;
    nnet::quantizer_1<q_dense_57_relu_t, quantizer_1_t>(layer7_out, layer29_out);
    [[intel::fpga_register]] q_add_21_t layer30_out;
    nnet::add<quantizer_t, quantizer_1_t, q_add_21_t, config30>(layer28_out, layer29_out, layer30_out);
    [[intel::fpga_register]] q_dense_58_iq_t layer31_out;
    nnet::q_dense_58_iq<q_add_21_t, q_dense_58_iq_t>(layer30_out, layer31_out);
    [[intel::fpga_register]] q_dense_58_t layer57_out;
    nnet::pointwise_conv_1d_cl<q_dense_58_iq_t, q_dense_58_t, config63>(layer31_out, layer57_out, w57, b57);
    [[intel::fpga_register]] q_dense_58_relu_t layer33_out;
    nnet::relu<q_dense_58_t, q_dense_58_relu_t, relu_config33>(layer57_out, layer33_out);
    [[intel::fpga_register]] q_dense_59_iq_t layer34_out;
    nnet::q_dense_59_iq<q_dense_58_relu_t, q_dense_59_iq_t>(layer33_out, layer34_out);
    [[intel::fpga_register]] q_dense_59_t layer58_out;
    nnet::pointwise_conv_1d_cl<q_dense_59_iq_t, q_dense_59_t, config64>(layer34_out, layer58_out, w58, b58);
    [[intel::fpga_register]] q_dense_59_relu_t layer36_out;
    nnet::relu<q_dense_59_t, q_dense_59_relu_t, relu_config36>(layer58_out, layer36_out);
    [[intel::fpga_register]] quantizer_2_t layer37_out;
    nnet::quantizer_2<q_dense_59_relu_t, quantizer_2_t>(layer36_out, layer37_out);
    [[intel::fpga_register]] quantizer_3_t layer38_out;
    nnet::quantizer_3<q_add_21_t, quantizer_3_t>(layer30_out, layer38_out);
    [[intel::fpga_register]] q_add_22_t layer39_out;
    nnet::add<quantizer_2_t, quantizer_3_t, q_add_22_t, config39>(layer37_out, layer38_out, layer39_out);
    [[intel::fpga_register]] q_dense_60_iq_t layer40_out;
    nnet::q_dense_60_iq<q_add_22_t, q_dense_60_iq_t>(layer39_out, layer40_out);
    [[intel::fpga_register]] q_dense_60_t layer59_out;
    nnet::pointwise_conv_1d_cl<q_dense_60_iq_t, q_dense_60_t, config65>(layer40_out, layer59_out, w59, b59);
    [[intel::fpga_register]] q_dense_60_relu_t layer42_out;
    nnet::relu<q_dense_60_t, q_dense_60_relu_t, relu_config42>(layer59_out, layer42_out);
    [[intel::fpga_register]] q_dense_61_iq_t layer43_out;
    nnet::q_dense_61_iq<q_dense_60_relu_t, q_dense_61_iq_t>(layer42_out, layer43_out);
    [[intel::fpga_register]] q_dense_61_t layer60_out;
    nnet::pointwise_conv_1d_cl<q_dense_61_iq_t, q_dense_61_t, config66>(layer43_out, layer60_out, w60, b60);
    [[intel::fpga_register]] q_dense_61_relu_t layer45_out;
    nnet::relu<q_dense_61_t, q_dense_61_relu_t, relu_config45>(layer60_out, layer45_out);
    [[intel::fpga_register]] quantizer_4_t layer46_out;
    nnet::quantizer_4<q_dense_61_relu_t, quantizer_4_t>(layer45_out, layer46_out);
    [[intel::fpga_register]] quantizer_5_t layer47_out;
    nnet::quantizer_5<q_add_22_t, quantizer_5_t>(layer39_out, layer47_out);
    [[intel::fpga_register]] q_add_23_t layer48_out;
    nnet::add<quantizer_4_t, quantizer_5_t, q_add_23_t, config48>(layer46_out, layer47_out, layer48_out);
    auto& layer49_out = layer48_out;
    [[intel::fpga_register]] q_dense_62_iq_t layer50_out;
    nnet::q_dense_62_iq<q_add_23_t, q_dense_62_iq_t>(layer49_out, layer50_out);
    [[intel::fpga_register]] q_dense_62_t layer51_out;
    nnet::dense_resource<q_dense_62_iq_t, q_dense_62_t, config51>(layer50_out, layer51_out, w51, b51);
    [[intel::fpga_register]] q_dense_62_relu_t layer52_out;
    nnet::relu<q_dense_62_t, q_dense_62_relu_t, relu_config52>(layer51_out, layer52_out);
    [[intel::fpga_register]] q_dense_63_iq_t layer53_out;
    nnet::q_dense_63_iq<q_dense_62_relu_t, q_dense_63_iq_t>(layer52_out, layer53_out);
    [[intel::fpga_register]] result_t layer54_out;
    nnet::dense_resource<q_dense_63_iq_t, result_t, config54>(layer53_out, layer54_out, w54, b54);

    // hls-fpga-machine-learning return
    Layer54OutPipe::write(layer54_out);
}

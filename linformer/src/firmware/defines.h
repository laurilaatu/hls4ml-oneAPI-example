#ifndef DEFINES_H_
#define DEFINES_H_

#include <sycl/ext/intel/ac_types/ac_fixed.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

// Include nnet::array - a custom array-like struct, mainly used with io_stream
#include "nnet_utils/nnet_types.h"

// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ac_fixed<16,8,true>, 192*1> input_layer_7_t;
typedef nnet::array<ac_fixed<15,8,true>, 192*1> q_dense_56_iq_t;
typedef ac_fixed<25,8,true> q_dense_56_accum_t;
typedef nnet::array<ac_fixed<16,8,true>, 1024*1> q_dense_56_t;
typedef nnet::array<ac_fixed<12,2,true>, 48*1> w55_t;
typedef nnet::array<ac_fixed<12,2,true>, 16*1> b55_t;
typedef nnet::array<ac_fixed<16,8,true>, 1024*1> q_dense_56_relu_t;
typedef ac_fixed<18,8,true> q_dense_56_relu_table_t;
typedef nnet::array<ac_fixed<27,20,true>, 1024*1> q_dense_57_iq_t;
typedef ac_fixed<35,22,true> q_dense_57_accum_t;
typedef nnet::array<ac_fixed<30,22,true>, 1024*1> q_dense_57_t;
typedef nnet::array<ac_fixed<11,4,true>, 256*1> w56_t;
typedef nnet::array<ac_fixed<13,2,true>, 16*1> b56_t;
typedef nnet::array<ac_fixed<30,22,true>, 1024*1> q_dense_57_relu_t;
typedef ac_fixed<18,8,true> q_dense_57_relu_table_t;
typedef nnet::array<ac_fixed<6,2,false>, 1024*1> q_einsum_dense_iq_t;
typedef ac_fixed<10,5,true> q_einsum_dense_accum_t;
typedef nnet::array<ac_fixed<9,5,true>, 32*1> q_einsum_dense_t;
typedef nnet::array<ac_fixed<6,3,true>, 128*1> w9_t;
typedef nnet::array<ac_fixed<2,32,false>, 32*1> b9_t;
typedef nnet::array<ac_fixed<11,5,false>, 1024*1> q_einsum_dense_1_iq_t;
typedef ac_fixed<21,6,true> q_einsum_dense_1_accum_t;
typedef nnet::array<ac_fixed<13,6,true>, 32*1> q_einsum_dense_1_t;
typedef nnet::array<ac_fixed<10,1,true>, 128*1> w11_t;
typedef nnet::array<ac_fixed<2,32,false>, 32*1> b11_t;
typedef nnet::array<ac_fixed<3,1,false>, 1024*1> mha1_query_iq_t;
typedef ac_fixed<7,4,true> mha1_query_accum_t;
typedef nnet::array<ac_fixed<7,4,true>, 1024*1> mha1_query_t;
typedef nnet::array<ac_fixed<6,4,true>, 256*1> w13_t;
typedef nnet::array<ac_fixed<5,2,true>, 1024*1> b13_t;
typedef nnet::array<ac_fixed<5,3,true>, 1024*1> mha1_query_oq_t;
typedef nnet::array<ac_fixed<6,3,true>, 32*1> mha1_key_iq_t;
typedef ac_fixed<10,5,true> mha1_key_accum_t;
typedef nnet::array<ac_fixed<8,5,true>, 32*1> mha1_key_t;
typedef nnet::array<ac_fixed<7,3,true>, 256*1> w16_t;
typedef nnet::array<ac_fixed<2,32,false>, 32*1> b16_t;
typedef nnet::array<ac_fixed<5,3,true>, 32*1> mha1_key_oq_t;
typedef nnet::array<ac_fixed<11,5,true>, 32*1> mha1_value_iq_t;
typedef ac_fixed<21,6,true> mha1_value_accum_t;
typedef nnet::array<ac_fixed<14,6,true>, 32*1> mha1_value_t;
typedef nnet::array<ac_fixed<12,3,true>, 256*1> w19_t;
typedef nnet::array<ac_fixed<6,1,true>, 32*1> b19_t;
typedef nnet::array<ac_fixed<9,2,true>, 32*1> mha1_value_oq_t;
typedef ac_fixed<10,6,true> mha1_mha1_QK_accum_t;
typedef nnet::array<ac_fixed<10,6,true>, 128*1> mha1_mha1_QK_t;
typedef nnet::array<ac_fixed<8,5,true>, 128*1> mha1_q_softmax_iq_t;
typedef ac_fixed<5,1,false,AC_RND_CONV,AC_SAT> mha1_q_softmax_exp_table_t;
typedef ac_fixed<2,1,false,AC_RND_CONV,AC_SAT> mha1_q_softmax_inv_table_t;
typedef ac_fixed<2,-10,true,AC_RND,AC_SAT> mha1_q_softmax_inv_inp_t;
typedef ac_fixed<7,5,true,AC_RND,AC_SAT> mha1_q_softmax_inp_norm_t;
typedef ac_fixed<2,-9,true> mha1_q_softmax_accum_t;
typedef nnet::array<ac_fixed<7,2,false>, 128*1> mha1_q_softmax_t;
typedef ac_fixed<18,8,true> mha1_q_softmax_table_t;
typedef nnet::array<ac_fixed<5,1,false>, 128*1> mha1_q_softmax_oq_t;
typedef ac_fixed<14,3,true> mha1_mha1_aV_accum_t;
typedef nnet::array<ac_fixed<10,3,true>, 1024*1> mha1_mha1_aV_t;
typedef nnet::array<ac_fixed<8,2,true>, 1024*1> mha1_attention_output_iq_t;
typedef ac_fixed<14,4,true> mha1_attention_output_accum_t;
typedef nnet::array<ac_fixed<12,4,true>, 1024*1> mha1_attention_output_t;
typedef nnet::array<ac_fixed<7,2,true>, 256*1> w27_t;
typedef nnet::array<ac_fixed<5,1,true>, 1024*1> b27_t;
typedef nnet::array<ac_fixed<19,12,true>, 1024*1> quantizer_t;
typedef nnet::array<ac_fixed<20,13,true>, 1024*1> quantizer_1_t;
typedef nnet::array<ac_fixed<20,14,true>, 1024*1> q_add_21_t;
typedef nnet::array<ac_fixed<19,13,true>, 1024*1> q_dense_58_iq_t;
typedef ac_fixed<25,17,true> q_dense_58_accum_t;
typedef nnet::array<ac_fixed<22,17,true>, 2048*1> q_dense_58_t;
typedef nnet::array<ac_fixed<8,5,true>, 512*1> w57_t;
typedef nnet::array<ac_fixed<4,2,true>, 32*1> b57_t;
typedef nnet::array<ac_fixed<20,15,true>, 2048*1> q_dense_58_relu_t;
typedef ac_fixed<18,8,true> q_dense_58_relu_table_t;
typedef nnet::array<ac_fixed<19,15,true>, 2048*1> q_dense_59_iq_t;
typedef ac_fixed<25,19,true> q_dense_59_accum_t;
typedef nnet::array<ac_fixed<23,19,true>, 1024*1> q_dense_59_t;
typedef nnet::array<ac_fixed<7,4,true>, 512*1> w58_t;
typedef nnet::array<ac_fixed<3,3,true>, 16*1> b58_t;
typedef nnet::array<ac_fixed<19,15,true>, 1024*1> q_dense_59_relu_t;
typedef ac_fixed<18,8,true> q_dense_59_relu_table_t;
typedef nnet::array<ac_fixed<19,15,true>, 1024*1> quantizer_2_t;
typedef nnet::array<ac_fixed<20,15,true>, 1024*1> quantizer_3_t;
typedef nnet::array<ac_fixed<21,16,true>, 1024*1> q_add_22_t;
typedef nnet::array<ac_fixed<19,16,true>, 1024*1> q_dense_60_iq_t;
typedef ac_fixed<20,18,true> q_dense_60_accum_t;
typedef nnet::array<ac_fixed<19,18,true>, 2048*1> q_dense_60_t;
typedef nnet::array<ac_fixed<4,4,true>, 512*1> w59_t;
typedef nnet::array<ac_fixed<3,3,true>, 32*1> b59_t;
typedef nnet::array<ac_fixed<17,16,true>, 2048*1> q_dense_60_relu_t;
typedef ac_fixed<18,8,true> q_dense_60_relu_table_t;
typedef nnet::array<ac_fixed<16,16,true>, 2048*1> q_dense_61_iq_t;
typedef ac_fixed<19,15,true> q_dense_61_accum_t;
typedef nnet::array<ac_fixed<19,15,true>, 1024*1> q_dense_61_t;
typedef nnet::array<ac_fixed<4,0,false>, 512*1> w60_t;
typedef nnet::array<ac_fixed<3,3,true>, 16*1> b60_t;
typedef nnet::array<ac_fixed<19,15,true>, 1024*1> q_dense_61_relu_t;
typedef ac_fixed<18,8,true> q_dense_61_relu_table_t;
typedef nnet::array<ac_fixed<19,16,true>, 1024*1> quantizer_4_t;
typedef nnet::array<ac_fixed<20,16,true>, 1024*1> quantizer_5_t;
typedef nnet::array<ac_fixed<16,12,true>, 1024*1> q_add_23_t;
typedef nnet::array<ac_fixed<17,13,true>, 1024*1> q_dense_62_iq_t;
typedef ac_fixed<23,15,true> q_dense_62_accum_t;
typedef nnet::array<ac_fixed<18,15,true>, 16*1> q_dense_62_t;
typedef nnet::array<ac_fixed<10,4,true>, 16384*1> w51_t;
typedef nnet::array<ac_fixed<2,32,false>, 16*1> b51_t;
typedef ac_int<1, false> layer51_index;
typedef nnet::array<ac_fixed<14,11,true>, 16*1> q_dense_62_relu_t;
typedef ac_fixed<18,8,true> q_dense_62_relu_table_t;
typedef nnet::array<ac_fixed<13,11,true>, 16*1> q_dense_63_iq_t;
typedef ac_fixed<28,10,true> q_dense_63_accum_t;
typedef nnet::array<ac_fixed<28,10,true>, 5*1> result_t;
typedef nnet::array<ac_fixed<18,1,true>, 80*1> w54_t;
typedef nnet::array<ac_fixed<3,1,true>, 5*1> b54_t;
typedef ac_int<1, false> layer54_index;

#define DIV_ROUNDUP(n, d) ((n + d - 1) / d)
#define MIN(n, d) (n > d ? d : n)
#define MAX(n, d) (n < d ? d : n)

#endif

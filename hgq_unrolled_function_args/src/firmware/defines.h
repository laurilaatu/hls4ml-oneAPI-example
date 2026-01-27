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
typedef nnet::array<ac_fixed<5,2,true>, 4*1> input_layer_t;
typedef ac_fixed<19,3,true> q_dense_accum_t;
typedef nnet::array<ac_fixed<6,3,true>, 64*1> q_dense_t;
typedef nnet::array<ac_fixed<13,0,true>, 256*1> w3_t;
typedef nnet::array<ac_fixed<12,-3,true>, 64*1> b3_t;
typedef ac_int<1, false> layer3_index;
typedef nnet::array<ac_fixed<3,0,false>, 64*1> q_dense_relu_t;
typedef ac_fixed<18,8,true> q_dense_relu_table_t;
typedef nnet::array<ac_fixed<3,0,false>, 64*1> q_dense_1_iq_t;
typedef ac_fixed<25,3,true> q_dense_1_accum_t;
typedef nnet::array<ac_fixed<6,3,true>, 64*1> q_dense_1_t;
typedef nnet::array<ac_fixed<19,0,true>, 4096*1> w6_t;
typedef nnet::array<ac_fixed<14,-2,true>, 64*1> b6_t;
typedef ac_int<1, false> layer6_index;
typedef nnet::array<ac_fixed<3,0,false>, 64*1> q_dense_1_relu_t;
typedef ac_fixed<18,8,true> q_dense_1_relu_table_t;
typedef nnet::array<ac_fixed<3,0,false>, 64*1> q_dense_2_iq_t;
typedef ac_fixed<21,2,true> q_dense_2_accum_t;
typedef nnet::array<ac_fixed<5,2,true>, 64*1> q_dense_2_t;
typedef nnet::array<ac_fixed<19,0,true>, 4096*1> w9_t;
typedef nnet::array<ac_fixed<13,-2,true>, 64*1> b9_t;
typedef ac_int<1, false> layer9_index;
typedef nnet::array<ac_fixed<3,0,false>, 64*1> q_dense_2_relu_t;
typedef ac_fixed<18,8,true> q_dense_2_relu_table_t;
typedef nnet::array<ac_fixed<3,0,false>, 64*1> q_dense_3_iq_t;
typedef ac_fixed<17,2,true> q_dense_3_accum_t;
typedef nnet::array<ac_fixed<17,2,true>, 1*1> result_t;
typedef nnet::array<ac_fixed<12,0,true>, 64*1> w12_t;
typedef nnet::array<ac_fixed<5,-4,true>, 1*1> b12_t;
typedef ac_int<1, false> layer12_index;

#define DIV_ROUNDUP(n, d) ((n + d - 1) / d)
#define MIN(n, d) (n > d ? d : n)
#define MAX(n, d) (n < d ? d : n)

#endif

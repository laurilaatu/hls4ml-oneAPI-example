#ifndef DEFINES_H_
#define DEFINES_H_

#include <sycl/ext/intel/ac_types/ac_fixed.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

// Include nnet::array - a custom array-like struct, mainly used with io_stream
#include "nnet_utils/nnet_types.h"

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 5
#define N_INPUT_2_1 1
#define N_OUTPUTS_2 5
#define N_FILT_2 4
#define N_OUTPUTS_2 5
#define N_FILT_2 4
#define N_TIME_STEPS_4 5
#define N_OUT_4 16
#define N_OUTPUTS_7 5
#define N_FILT_7 4

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ac_fixed<12,6,true>, 1*1> input_t;
typedef ac_fixed<12,6,true> model_default_t;
typedef nnet::array<ac_fixed<28,16,true>, 4*1> conv1d_3_result_t;
typedef nnet::array<ac_fixed<12,6,true>, 20*1> w2_t;
typedef nnet::array<ac_fixed<12,6,true>, 4*1> b2_t;
typedef nnet::array<ac_fixed<12,6,true>, 4*1> layer3_t;
typedef ac_fixed<18,8,true> conv1d_3_relu_table_t;
typedef nnet::array<ac_fixed<12,6,true>, 16*1> layer4_t;
typedef nnet::array<ac_fixed<12,6,true>, 192*1> w4_t;
typedef nnet::array<ac_fixed<12,6,true>, 768*1> wr4_t;
typedef nnet::array<ac_fixed<12,6,true>, 48*1> b4_t;
typedef nnet::array<ac_fixed<12,6,true>, 48*1> br4_t;
typedef ac_fixed<18,8,true> gru_3_table_t;
typedef nnet::array<ac_fixed<29,17,true>, 4*1> result_t;
typedef nnet::array<ac_fixed<12,6,true>, 64*1> w7_t;
typedef nnet::array<ac_fixed<12,6,true>, 4*1> b7_t;

#define DIV_ROUNDUP(n, d) ((n + d - 1) / d)
#define MIN(n, d) (n > d ? d : n)
#define MAX(n, d) (n < d ? d : n)

#endif

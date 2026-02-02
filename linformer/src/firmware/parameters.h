#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "defines.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"

// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_conv1d.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_einsum.h"
#include "nnet_utils/nnet_einsum_dense.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_stream.h"
#include "nnet_utils/nnet_stream.h"

// hls-fpga-machine-learning insert layer-config
struct config61_mult : nnet::dense_config {
    static const unsigned n_in = 3;
    static const unsigned n_out = 16;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 1;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_dense_56_accum_t accum_t;
    typedef b55_t bias_t;
    typedef w55_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config61 : nnet::conv1d_config {
    static const unsigned in_width = 64;
    static const unsigned n_chan = 3;

    static const unsigned filt_width = 1;
    static const unsigned impl_filt_width = 1;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 16;
    static const unsigned out_width = 64;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 1;
    static const unsigned parallelization_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::im2col;

    typedef q_dense_56_accum_t accum_t;
    typedef b55_t bias_t;
    typedef w55_t weight_t;
    typedef config61_mult mult_config;
};

struct relu_config4 : nnet::activ_config {
    static constexpr unsigned n_in = 1024;
    static constexpr unsigned table_size = 33554432;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned reuse_factor = 1;
    typedef q_dense_56_relu_table_t table_t;
};

struct config62_mult : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 16;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 1;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_dense_57_accum_t accum_t;
    typedef b56_t bias_t;
    typedef w56_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config62 : nnet::conv1d_config {
    static const unsigned in_width = 64;
    static const unsigned n_chan = 16;

    static const unsigned filt_width = 1;
    static const unsigned impl_filt_width = 1;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 16;
    static const unsigned out_width = 64;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 1;
    static const unsigned parallelization_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::im2col;

    typedef q_dense_57_accum_t accum_t;
    typedef b56_t bias_t;
    typedef w56_t weight_t;
    typedef config62_mult mult_config;
};

struct relu_config7 : nnet::activ_config {
    static constexpr unsigned n_in = 1024;
    static constexpr unsigned table_size = 34359738368;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned reuse_factor = 1;
    typedef q_dense_57_relu_table_t table_t;
};

struct config9_tpose_inp : nnet::transpose_config {
    static constexpr unsigned dims = 2;
    static constexpr unsigned N = 1024;
    static constexpr std::array<unsigned, dims> from_shape = {64, 16};
    static constexpr std::array<unsigned, dims> to_shape = {16, 64};
    static constexpr std::array<unsigned, dims> perm = {1, 0};
    static constexpr std::array<unsigned, dims> perm_strides = {1, 16};
};


struct config9_tpose_out : nnet::transpose_config {
    static constexpr unsigned dims = 2;
    static constexpr unsigned N = 32;
    static constexpr std::array<unsigned, dims> from_shape = {16, 2};
    static constexpr std::array<unsigned, dims> to_shape = {2, 16};
    static constexpr std::array<unsigned, dims> perm = {1, 0};
    static constexpr std::array<unsigned, dims> perm_strides = {1, 2};
};


struct config9_dense : nnet::dense_config {
    static constexpr unsigned n_in = 64;
    static constexpr unsigned n_out = 2;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned n_zeros = 72;
    static constexpr unsigned n_nonzeros = 56;
    static constexpr bool store_weights_in_bram = false;

    static constexpr unsigned rf_pad = 0;
    static constexpr unsigned bf_pad = 0;

    static constexpr unsigned reuse_factor = 1;
    static constexpr unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static constexpr unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static constexpr unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static constexpr unsigned block_factor_rounded = block_factor + bf_pad;
    static constexpr unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static constexpr unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static constexpr unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_einsum_dense_accum_t accum_t;
    typedef b9_t bias_t;
    typedef w9_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



struct config9 {
    typedef config9_tpose_inp tpose_inp_conf;
    typedef config9_tpose_out tpose_out_conf;

    typedef q_einsum_dense_accum_t accum_t;
    typedef w9_t weight_t;
    typedef b9_t bias_t;

    typedef config9_dense dense_conf;

    // Layer Sizes
    static constexpr unsigned n_free_data = 16;
    static constexpr unsigned n_free_kernel = 2;
    static constexpr unsigned n_contract = 64;
    static constexpr unsigned n_inplace = 1;

    // Resource reuse info
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned reuse_factor = 1;
    static constexpr unsigned parallelization_factor = 16; // Only useful when n_inplace > 1
};

struct config11_tpose_inp : nnet::transpose_config {
    static constexpr unsigned dims = 2;
    static constexpr unsigned N = 1024;
    static constexpr std::array<unsigned, dims> from_shape = {64, 16};
    static constexpr std::array<unsigned, dims> to_shape = {16, 64};
    static constexpr std::array<unsigned, dims> perm = {1, 0};
    static constexpr std::array<unsigned, dims> perm_strides = {1, 16};
};


struct config11_tpose_out : nnet::transpose_config {
    static constexpr unsigned dims = 2;
    static constexpr unsigned N = 32;
    static constexpr std::array<unsigned, dims> from_shape = {16, 2};
    static constexpr std::array<unsigned, dims> to_shape = {2, 16};
    static constexpr std::array<unsigned, dims> perm = {1, 0};
    static constexpr std::array<unsigned, dims> perm_strides = {1, 2};
};


struct config11_dense : nnet::dense_config {
    static constexpr unsigned n_in = 64;
    static constexpr unsigned n_out = 2;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned n_zeros = 19;
    static constexpr unsigned n_nonzeros = 109;
    static constexpr bool store_weights_in_bram = false;

    static constexpr unsigned rf_pad = 0;
    static constexpr unsigned bf_pad = 0;

    static constexpr unsigned reuse_factor = 1;
    static constexpr unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static constexpr unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static constexpr unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static constexpr unsigned block_factor_rounded = block_factor + bf_pad;
    static constexpr unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static constexpr unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static constexpr unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_einsum_dense_1_accum_t accum_t;
    typedef b11_t bias_t;
    typedef w11_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



struct config11 {
    typedef config11_tpose_inp tpose_inp_conf;
    typedef config11_tpose_out tpose_out_conf;

    typedef q_einsum_dense_1_accum_t accum_t;
    typedef w11_t weight_t;
    typedef b11_t bias_t;

    typedef config11_dense dense_conf;

    // Layer Sizes
    static constexpr unsigned n_free_data = 16;
    static constexpr unsigned n_free_kernel = 2;
    static constexpr unsigned n_contract = 64;
    static constexpr unsigned n_inplace = 1;

    // Resource reuse info
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned reuse_factor = 1;
    static constexpr unsigned parallelization_factor = 16; // Only useful when n_inplace > 1
};

struct config13_tpose_inp : nnet::transpose_config {
    static constexpr unsigned dims = 2;
    static constexpr unsigned N = 1024;
    static constexpr std::array<unsigned, dims> from_shape = {64, 16};
    static constexpr std::array<unsigned, dims> to_shape = {64, 16};
    static constexpr std::array<unsigned, dims> perm = {0, 1};
    static constexpr std::array<unsigned, dims> perm_strides = {16, 1};
};


struct config13_tpose_out : nnet::transpose_config {
    static constexpr unsigned dims = 3;
    static constexpr unsigned N = 1024;
    static constexpr std::array<unsigned, dims> from_shape = {64, 1, 16};
    static constexpr std::array<unsigned, dims> to_shape = {64, 1, 16};
    static constexpr std::array<unsigned, dims> perm = {0, 1, 2};
    static constexpr std::array<unsigned, dims> perm_strides = {16, 16, 1};
};


struct config13_dense : nnet::dense_config {
    static constexpr unsigned n_in = 16;
    static constexpr unsigned n_out = 16;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned n_zeros = 142;
    static constexpr unsigned n_nonzeros = 114;
    static constexpr bool store_weights_in_bram = false;

    static constexpr unsigned rf_pad = 0;
    static constexpr unsigned bf_pad = 0;

    static constexpr unsigned reuse_factor = 1;
    static constexpr unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static constexpr unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static constexpr unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static constexpr unsigned block_factor_rounded = block_factor + bf_pad;
    static constexpr unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static constexpr unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static constexpr unsigned multiplier_scale = multiplier_limit/n_out;

    typedef mha1_query_accum_t accum_t;
    typedef b13_t bias_t;
    typedef w13_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



struct config13 {
    typedef config13_tpose_inp tpose_inp_conf;
    typedef config13_tpose_out tpose_out_conf;

    typedef mha1_query_accum_t accum_t;
    typedef w13_t weight_t;
    typedef b13_t bias_t;

    typedef config13_dense dense_conf;

    // Layer Sizes
    static constexpr unsigned n_free_data = 64;
    static constexpr unsigned n_free_kernel = 16;
    static constexpr unsigned n_contract = 16;
    static constexpr unsigned n_inplace = 1;

    // Resource reuse info
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned reuse_factor = 1;
    static constexpr unsigned parallelization_factor = 64; // Only useful when n_inplace > 1
};

struct config16_tpose_inp : nnet::transpose_config {
    static constexpr unsigned dims = 2;
    static constexpr unsigned N = 32;
    static constexpr std::array<unsigned, dims> from_shape = {2, 16};
    static constexpr std::array<unsigned, dims> to_shape = {2, 16};
    static constexpr std::array<unsigned, dims> perm = {0, 1};
    static constexpr std::array<unsigned, dims> perm_strides = {16, 1};
};


struct config16_tpose_out : nnet::transpose_config {
    static constexpr unsigned dims = 3;
    static constexpr unsigned N = 32;
    static constexpr std::array<unsigned, dims> from_shape = {2, 1, 16};
    static constexpr std::array<unsigned, dims> to_shape = {2, 1, 16};
    static constexpr std::array<unsigned, dims> perm = {0, 1, 2};
    static constexpr std::array<unsigned, dims> perm_strides = {16, 16, 1};
};


struct config16_dense : nnet::dense_config {
    static constexpr unsigned n_in = 16;
    static constexpr unsigned n_out = 16;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned n_zeros = 169;
    static constexpr unsigned n_nonzeros = 87;
    static constexpr bool store_weights_in_bram = false;

    static constexpr unsigned rf_pad = 0;
    static constexpr unsigned bf_pad = 0;

    static constexpr unsigned reuse_factor = 1;
    static constexpr unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static constexpr unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static constexpr unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static constexpr unsigned block_factor_rounded = block_factor + bf_pad;
    static constexpr unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static constexpr unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static constexpr unsigned multiplier_scale = multiplier_limit/n_out;

    typedef mha1_key_accum_t accum_t;
    typedef b16_t bias_t;
    typedef w16_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



struct config16 {
    typedef config16_tpose_inp tpose_inp_conf;
    typedef config16_tpose_out tpose_out_conf;

    typedef mha1_key_accum_t accum_t;
    typedef w16_t weight_t;
    typedef b16_t bias_t;

    typedef config16_dense dense_conf;

    // Layer Sizes
    static constexpr unsigned n_free_data = 2;
    static constexpr unsigned n_free_kernel = 16;
    static constexpr unsigned n_contract = 16;
    static constexpr unsigned n_inplace = 1;

    // Resource reuse info
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned reuse_factor = 1;
    static constexpr unsigned parallelization_factor = 2; // Only useful when n_inplace > 1
};

struct config19_tpose_inp : nnet::transpose_config {
    static constexpr unsigned dims = 2;
    static constexpr unsigned N = 32;
    static constexpr std::array<unsigned, dims> from_shape = {2, 16};
    static constexpr std::array<unsigned, dims> to_shape = {2, 16};
    static constexpr std::array<unsigned, dims> perm = {0, 1};
    static constexpr std::array<unsigned, dims> perm_strides = {16, 1};
};


struct config19_tpose_out : nnet::transpose_config {
    static constexpr unsigned dims = 3;
    static constexpr unsigned N = 32;
    static constexpr std::array<unsigned, dims> from_shape = {2, 1, 16};
    static constexpr std::array<unsigned, dims> to_shape = {2, 1, 16};
    static constexpr std::array<unsigned, dims> perm = {0, 1, 2};
    static constexpr std::array<unsigned, dims> perm_strides = {16, 16, 1};
};


struct config19_dense : nnet::dense_config {
    static constexpr unsigned n_in = 16;
    static constexpr unsigned n_out = 16;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned n_zeros = 84;
    static constexpr unsigned n_nonzeros = 172;
    static constexpr bool store_weights_in_bram = false;

    static constexpr unsigned rf_pad = 0;
    static constexpr unsigned bf_pad = 0;

    static constexpr unsigned reuse_factor = 1;
    static constexpr unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static constexpr unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static constexpr unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static constexpr unsigned block_factor_rounded = block_factor + bf_pad;
    static constexpr unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static constexpr unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static constexpr unsigned multiplier_scale = multiplier_limit/n_out;

    typedef mha1_value_accum_t accum_t;
    typedef b19_t bias_t;
    typedef w19_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



struct config19 {
    typedef config19_tpose_inp tpose_inp_conf;
    typedef config19_tpose_out tpose_out_conf;

    typedef mha1_value_accum_t accum_t;
    typedef w19_t weight_t;
    typedef b19_t bias_t;

    typedef config19_dense dense_conf;

    // Layer Sizes
    static constexpr unsigned n_free_data = 2;
    static constexpr unsigned n_free_kernel = 16;
    static constexpr unsigned n_contract = 16;
    static constexpr unsigned n_inplace = 1;

    // Resource reuse info
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned reuse_factor = 1;
    static constexpr unsigned parallelization_factor = 2; // Only useful when n_inplace > 1
};

struct config21_tpose_inp0 : nnet::transpose_config {
    static constexpr unsigned dims = 3;
    static constexpr unsigned N = 32;
    static constexpr std::array<unsigned, dims> from_shape = {2, 1, 16};
    static constexpr std::array<unsigned, dims> to_shape = {1, 2, 16};
    static constexpr std::array<unsigned, dims> perm = {1, 0, 2};
    static constexpr std::array<unsigned, dims> perm_strides = {16, 16, 1};
};


struct config21_tpose_inp1 : nnet::transpose_config {
    static constexpr unsigned dims = 3;
    static constexpr unsigned N = 1024;
    static constexpr std::array<unsigned, dims> from_shape = {64, 1, 16};
    static constexpr std::array<unsigned, dims> to_shape = {1, 64, 16};
    static constexpr std::array<unsigned, dims> perm = {1, 0, 2};
    static constexpr std::array<unsigned, dims> perm_strides = {16, 16, 1};
};


struct config21_tpose_out : nnet::transpose_config {
    static constexpr unsigned dims = 3;
    static constexpr unsigned N = 128;
    static constexpr std::array<unsigned, dims> from_shape = {1, 2, 64};
    static constexpr std::array<unsigned, dims> to_shape = {1, 64, 2};
    static constexpr std::array<unsigned, dims> perm = {0, 2, 1};
    static constexpr std::array<unsigned, dims> perm_strides = {128, 1, 64};
};



struct config21 {
    typedef config21_tpose_inp0 tpose_inp0_config;
    typedef config21_tpose_inp1 tpose_inp1_config;
    typedef config21_tpose_out tpose_out_conf;

    typedef mha1_mha1_QK_accum_t accum_t;

    // Layer Sizes
    static const unsigned n_free0 = 2;
    static const unsigned n_free1 = 64;
    static const unsigned n_contract = 16;
    static const unsigned n_inplace = 1;

    // Resource reuse info
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = 2048;
    static const bool store_weights_in_bram = false; // NOT USED

    template <class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct softmax_config23 : nnet::activ_config {
    static const unsigned n_in = 128;
    static const unsigned n_slice = 2;
    static const unsigned n_outer = 64;
    static const unsigned n_inner = 1;
    static const unsigned parallelization_factor = 64;
    static const unsigned exp_table_size = 128;
    static const unsigned inv_table_size = 2;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned axis = -1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::stable;
    static constexpr float exp_scale = 0.25;
    typedef mha1_q_softmax_exp_table_t exp_table_t;
    typedef mha1_q_softmax_inv_table_t inv_table_t;
    typedef mha1_q_softmax_accum_t accum_t;
    typedef mha1_q_softmax_inv_inp_t inv_inp_t;
    typedef mha1_q_softmax_inp_norm_t inp_norm_t;
};

struct config25_tpose_inp0 : nnet::transpose_config {
    static constexpr unsigned dims = 3;
    static constexpr unsigned N = 128;
    static constexpr std::array<unsigned, dims> from_shape = {1, 64, 2};
    static constexpr std::array<unsigned, dims> to_shape = {1, 64, 2};
    static constexpr std::array<unsigned, dims> perm = {0, 1, 2};
    static constexpr std::array<unsigned, dims> perm_strides = {128, 2, 1};
};


struct config25_tpose_inp1 : nnet::transpose_config {
    static constexpr unsigned dims = 3;
    static constexpr unsigned N = 32;
    static constexpr std::array<unsigned, dims> from_shape = {2, 1, 16};
    static constexpr std::array<unsigned, dims> to_shape = {1, 16, 2};
    static constexpr std::array<unsigned, dims> perm = {1, 2, 0};
    static constexpr std::array<unsigned, dims> perm_strides = {16, 1, 16};
};


struct config25_tpose_out : nnet::transpose_config {
    static constexpr unsigned dims = 3;
    static constexpr unsigned N = 1024;
    static constexpr std::array<unsigned, dims> from_shape = {1, 64, 16};
    static constexpr std::array<unsigned, dims> to_shape = {64, 1, 16};
    static constexpr std::array<unsigned, dims> perm = {1, 0, 2};
    static constexpr std::array<unsigned, dims> perm_strides = {16, 1024, 1};
};



struct config25 {
    typedef config25_tpose_inp0 tpose_inp0_config;
    typedef config25_tpose_inp1 tpose_inp1_config;
    typedef config25_tpose_out tpose_out_conf;

    typedef mha1_mha1_aV_accum_t accum_t;

    // Layer Sizes
    static const unsigned n_free0 = 64;
    static const unsigned n_free1 = 16;
    static const unsigned n_contract = 2;
    static const unsigned n_inplace = 1;

    // Resource reuse info
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = 2048;
    static const bool store_weights_in_bram = false; // NOT USED

    template <class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config27_tpose_inp : nnet::transpose_config {
    static constexpr unsigned dims = 3;
    static constexpr unsigned N = 1024;
    static constexpr std::array<unsigned, dims> from_shape = {64, 1, 16};
    static constexpr std::array<unsigned, dims> to_shape = {64, 1, 16};
    static constexpr std::array<unsigned, dims> perm = {0, 1, 2};
    static constexpr std::array<unsigned, dims> perm_strides = {16, 16, 1};
};


struct config27_tpose_out : nnet::transpose_config {
    static constexpr unsigned dims = 2;
    static constexpr unsigned N = 1024;
    static constexpr std::array<unsigned, dims> from_shape = {64, 16};
    static constexpr std::array<unsigned, dims> to_shape = {64, 16};
    static constexpr std::array<unsigned, dims> perm = {0, 1};
    static constexpr std::array<unsigned, dims> perm_strides = {16, 1};
};


struct config27_dense : nnet::dense_config {
    static constexpr unsigned n_in = 16;
    static constexpr unsigned n_out = 16;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned n_zeros = 28;
    static constexpr unsigned n_nonzeros = 228;
    static constexpr bool store_weights_in_bram = false;

    static constexpr unsigned rf_pad = 0;
    static constexpr unsigned bf_pad = 0;

    static constexpr unsigned reuse_factor = 1;
    static constexpr unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static constexpr unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static constexpr unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static constexpr unsigned block_factor_rounded = block_factor + bf_pad;
    static constexpr unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static constexpr unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static constexpr unsigned multiplier_scale = multiplier_limit/n_out;

    typedef mha1_attention_output_accum_t accum_t;
    typedef b27_t bias_t;
    typedef w27_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



struct config27 {
    typedef config27_tpose_inp tpose_inp_conf;
    typedef config27_tpose_out tpose_out_conf;

    typedef mha1_attention_output_accum_t accum_t;
    typedef w27_t weight_t;
    typedef b27_t bias_t;

    typedef config27_dense dense_conf;

    // Layer Sizes
    static constexpr unsigned n_free_data = 64;
    static constexpr unsigned n_free_kernel = 16;
    static constexpr unsigned n_contract = 16;
    static constexpr unsigned n_inplace = 1;

    // Resource reuse info
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned reuse_factor = 1;
    static constexpr unsigned parallelization_factor = 64; // Only useful when n_inplace > 1
};

struct config30 : nnet::merge_config {
    static const unsigned n_elem = 64*16;
    static const unsigned reuse_factor = 1;
};

struct config63_mult : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 32;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 1;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_dense_58_accum_t accum_t;
    typedef b57_t bias_t;
    typedef w57_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config63 : nnet::conv1d_config {
    static const unsigned in_width = 64;
    static const unsigned n_chan = 16;

    static const unsigned filt_width = 1;
    static const unsigned impl_filt_width = 1;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 32;
    static const unsigned out_width = 64;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 1;
    static const unsigned parallelization_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::im2col;

    typedef q_dense_58_accum_t accum_t;
    typedef b57_t bias_t;
    typedef w57_t weight_t;
    typedef config63_mult mult_config;
};

struct relu_config33 : nnet::activ_config {
    static constexpr unsigned n_in = 2048;
    static constexpr unsigned table_size = 33554432;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned reuse_factor = 1;
    typedef q_dense_58_relu_table_t table_t;
};

struct config64_mult : nnet::dense_config {
    static const unsigned n_in = 32;
    static const unsigned n_out = 16;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 1;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_dense_59_accum_t accum_t;
    typedef b58_t bias_t;
    typedef w58_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config64 : nnet::conv1d_config {
    static const unsigned in_width = 64;
    static const unsigned n_chan = 32;

    static const unsigned filt_width = 1;
    static const unsigned impl_filt_width = 1;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 16;
    static const unsigned out_width = 64;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 1;
    static const unsigned parallelization_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::im2col;

    typedef q_dense_59_accum_t accum_t;
    typedef b58_t bias_t;
    typedef w58_t weight_t;
    typedef config64_mult mult_config;
};

struct relu_config36 : nnet::activ_config {
    static constexpr unsigned n_in = 1024;
    static constexpr unsigned table_size = 33554432;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned reuse_factor = 1;
    typedef q_dense_59_relu_table_t table_t;
};

struct config39 : nnet::merge_config {
    static const unsigned n_elem = 64*16;
    static const unsigned reuse_factor = 1;
};

struct config65_mult : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 32;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 1;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_dense_60_accum_t accum_t;
    typedef b59_t bias_t;
    typedef w59_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config65 : nnet::conv1d_config {
    static const unsigned in_width = 64;
    static const unsigned n_chan = 16;

    static const unsigned filt_width = 1;
    static const unsigned impl_filt_width = 1;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 32;
    static const unsigned out_width = 64;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 1;
    static const unsigned parallelization_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::im2col;

    typedef q_dense_60_accum_t accum_t;
    typedef b59_t bias_t;
    typedef w59_t weight_t;
    typedef config65_mult mult_config;
};

struct relu_config42 : nnet::activ_config {
    static constexpr unsigned n_in = 2048;
    static constexpr unsigned table_size = 1048576;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned reuse_factor = 1;
    typedef q_dense_60_relu_table_t table_t;
};

struct config66_mult : nnet::dense_config {
    static const unsigned n_in = 32;
    static const unsigned n_out = 16;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 1;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_dense_61_accum_t accum_t;
    typedef b60_t bias_t;
    typedef w60_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config66 : nnet::conv1d_config {
    static const unsigned in_width = 64;
    static const unsigned n_chan = 32;

    static const unsigned filt_width = 1;
    static const unsigned impl_filt_width = 1;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 16;
    static const unsigned out_width = 64;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 1;
    static const unsigned parallelization_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::im2col;

    typedef q_dense_61_accum_t accum_t;
    typedef b60_t bias_t;
    typedef w60_t weight_t;
    typedef config66_mult mult_config;
};

struct relu_config45 : nnet::activ_config {
    static constexpr unsigned n_in = 1024;
    static constexpr unsigned table_size = 524288;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned reuse_factor = 1;
    typedef q_dense_61_relu_table_t table_t;
};

struct config48 : nnet::merge_config {
    static const unsigned n_elem = 64*16;
    static const unsigned reuse_factor = 1;
};

struct config51 : nnet::dense_config {

    static constexpr unsigned n_in = 1024;
    static constexpr unsigned n_out = 16;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned n_zeros = 12550;
    static constexpr unsigned n_nonzeros = 3834;
    static constexpr bool store_weights_in_bram = false;

    static constexpr unsigned rf_pad = 0;
    static constexpr unsigned bf_pad = 0;

    static constexpr unsigned reuse_factor = 1;
    static constexpr unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static constexpr unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static constexpr unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static constexpr unsigned block_factor_rounded = block_factor + bf_pad;
    static constexpr unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static constexpr unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static constexpr unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_dense_62_accum_t accum_t;
    typedef b51_t bias_t;
    typedef w51_t weight_t;
    typedef layer51_index index_t;

    static constexpr weight_t weights = w51;
    static constexpr bias_t biases = b51;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct relu_config52 : nnet::activ_config {
    static constexpr unsigned n_in = 16;
    static constexpr unsigned table_size = 8388608;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned reuse_factor = 1;
    typedef q_dense_62_relu_table_t table_t;
};

struct config54 : nnet::dense_config {

    static constexpr unsigned n_in = 16;
    static constexpr unsigned n_out = 5;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned n_zeros = 1;
    static constexpr unsigned n_nonzeros = 79;
    static constexpr bool store_weights_in_bram = false;

    static constexpr unsigned rf_pad = 0;
    static constexpr unsigned bf_pad = 0;

    static constexpr unsigned reuse_factor = 1;
    static constexpr unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static constexpr unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static constexpr unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static constexpr unsigned block_factor_rounded = block_factor + bf_pad;
    static constexpr unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static constexpr unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static constexpr unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_dense_63_accum_t accum_t;
    typedef b54_t bias_t;
    typedef w54_t weight_t;
    typedef layer54_index index_t;

    static constexpr weight_t weights = w54;
    static constexpr bias_t biases = b54;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};


#endif

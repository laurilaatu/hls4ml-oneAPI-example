#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "defines.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"

// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_stream.h"

// hls-fpga-machine-learning insert layer-config
struct config3 : nnet::dense_config {
    static constexpr unsigned n_in = 4;
    static constexpr unsigned n_out = 64;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned n_zeros = 0;
    static constexpr unsigned n_nonzeros = 256;
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

    typedef q_dense_accum_t accum_t;
    typedef b3_t bias_t;
    typedef w3_t weight_t;
    typedef layer3_index index_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct relu_config4 : nnet::activ_config {
    static constexpr unsigned n_in = 64;
    static constexpr unsigned table_size = 524288;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned reuse_factor = 1;
    typedef q_dense_relu_table_t table_t;
};

struct config6 : nnet::dense_config {
    static constexpr unsigned n_in = 64;
    static constexpr unsigned n_out = 64;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned n_zeros = 0;
    static constexpr unsigned n_nonzeros = 4096;
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

    typedef q_dense_1_accum_t accum_t;
    typedef b6_t bias_t;
    typedef w6_t weight_t;
    typedef layer6_index index_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct relu_config7 : nnet::activ_config {
    static constexpr unsigned n_in = 64;
    static constexpr unsigned table_size = 33554432;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned reuse_factor = 1;
    typedef q_dense_1_relu_table_t table_t;
};

struct config9 : nnet::dense_config {
    static constexpr unsigned n_in = 64;
    static constexpr unsigned n_out = 64;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned n_zeros = 0;
    static constexpr unsigned n_nonzeros = 4096;
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

    typedef q_dense_2_accum_t accum_t;
    typedef b9_t bias_t;
    typedef w9_t weight_t;
    typedef layer9_index index_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct relu_config10 : nnet::activ_config {
    static constexpr unsigned n_in = 64;
    static constexpr unsigned table_size = 2097152;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned reuse_factor = 1;
    typedef q_dense_2_relu_table_t table_t;
};

struct config12 : nnet::dense_config {
    static constexpr unsigned n_in = 64;
    static constexpr unsigned n_out = 1;
    static constexpr unsigned io_type = nnet::io_parallel;
    static constexpr unsigned n_zeros = 0;
    static constexpr unsigned n_nonzeros = 64;
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

    typedef q_dense_3_accum_t accum_t;
    typedef b12_t bias_t;
    typedef w12_t weight_t;
    typedef layer12_index index_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};


#endif

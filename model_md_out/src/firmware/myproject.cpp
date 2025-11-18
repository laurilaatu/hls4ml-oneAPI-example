#include "myproject.h"
#include "parameters.h"
#include <sycl/ext/intel/experimental/task_sequence.hpp>

// hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w4.h"
#include "weights/wr4.h"
#include "weights/b4.h"
#include "weights/br4.h"
#include "weights/w5.h"
#include "weights/wr5.h"
#include "weights/b5.h"
#include "weights/br5.h"
#include "weights/w6.h"
#include "weights/b6.h"

// The inter-task pipes need to be declared in the global scope
// hls-fpga-machine-learning insert inter-task pipes
class Layer2OutPipeID;
using Layer2OutPipe = sycl::ext::intel::experimental::pipe<Layer2OutPipeID, conv1d_result_t, 5>;
class Layer3OutPipeID;
using Layer3OutPipe = sycl::ext::intel::experimental::pipe<Layer3OutPipeID, layer3_t, 5>;
class Layer4OutPipeID;
using Layer4OutPipe = sycl::ext::intel::experimental::pipe<Layer4OutPipeID, layer4_t, 5>;
class Layer5OutPipeID;
using Layer5OutPipe = sycl::ext::intel::experimental::pipe<Layer5OutPipeID, layer5_t, 1>;

using sycl::ext::intel::experimental::task_sequence;

void Myproject::operator()() const {
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning read in

    // hls-fpga-machine-learning declare task sequences
    task_sequence<nnet::conv_1d_cl_stream<Conv1DInputPipe, Layer2OutPipe, config2>> conv1d;
    task_sequence<nnet::relu_stream<Layer2OutPipe, Layer3OutPipe, relu_config3>> conv1d_relu;
    task_sequence<nnet::gru_stream<Layer3OutPipe, Layer4OutPipe, config4>> gru;
    task_sequence<nnet::gru_stream<Layer4OutPipe, Layer5OutPipe, config5>> gru_1;
    task_sequence<nnet::dense_resource_stream<Layer5OutPipe, Layer6OutPipe, config6>> dense;

    // hls-fpga-machine-learning insert layers

    conv1d.async(w2, b2);
    conv1d_relu.async();
    gru.async(w4, wr4, b4, br4);
    gru_1.async(w5, wr5, b5, br5);
    dense.async(w6, b6);

    // hls-fpga-machine-learning return
}

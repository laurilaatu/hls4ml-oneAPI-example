#include "myproject.h"
#include "parameters.h"
#include <sycl/ext/intel/experimental/task_sequence.hpp>

// hls-fpga-machine-learning insert weights



// The inter-task pipes need to be declared in the global scope
// hls-fpga-machine-learning insert inter-task pipes
class Layer2OutPipeID;
using Layer2OutPipe = sycl::ext::intel::experimental::pipe<Layer2OutPipeID, conv1d_3_result_t, 5>;
class Layer3OutPipeID;
using Layer3OutPipe = sycl::ext::intel::experimental::pipe<Layer3OutPipeID, layer3_t, 5>;
class Layer4OutPipeID;
using Layer4OutPipe = sycl::ext::intel::experimental::pipe<Layer4OutPipeID, layer4_t, 5>;

using sycl::ext::intel::experimental::task_sequence;

void Myproject::operator()() const {
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning read in

    // hls-fpga-machine-learning declare task sequences
    task_sequence<nnet::conv_1d_cl_stream<Conv1D3InputPipe, Layer2OutPipe, config2>> conv1d_3;
    task_sequence<nnet::relu_stream<Layer2OutPipe, Layer3OutPipe, relu_config3>> conv1d_3_relu;
    task_sequence<nnet::gru_stream<Layer3OutPipe, Layer4OutPipe, config4>> gru_3;
    task_sequence<nnet::conv_1d_cl_stream<Layer4OutPipe, Layer7OutPipe, config7>> dense_3;

    // hls-fpga-machine-learning insert layers

    conv1d_3.async();
    conv1d_3_relu.async();
    gru_3.async();
    dense_3.async();

    // hls-fpga-machine-learning return
}

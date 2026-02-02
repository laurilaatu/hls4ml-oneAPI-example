#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "defines.h"

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


// This file defines the interface to the kernel

// currently this is fixed
using PipeProps = decltype(sycl::ext::oneapi::experimental::properties(sycl::ext::intel::experimental::ready_latency<0>));

// Need to declare the input and output pipes

// hls-fpga-machine-learning insert inputs
class InputLayer7PipeID;
using InputLayer7Pipe = sycl::ext::intel::experimental::pipe<InputLayer7PipeID, input_layer_7_t, 0, PipeProps>;
// hls-fpga-machine-learning insert outputs
class Layer54OutPipeID;
using Layer54OutPipe = sycl::ext::intel::experimental::pipe<Layer54OutPipeID, result_t, 0, PipeProps>;

class MyprojectID;

struct Myproject {

    // kernel property method to config invocation interface
    auto get(sycl::ext::oneapi::experimental::properties_tag) {
        return sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::streaming_interface<>,
                                                           sycl::ext::intel::experimental::pipelined<>};
    }

    SYCL_EXTERNAL void operator()() const;
};

#endif

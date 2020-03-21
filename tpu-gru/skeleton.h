#include "flipflop_r941native.h"

typedef struct {
    //  Convolution layer
    const flappie_matrix conv_W;
    const flappie_matrix conv_b;
    int conv_stride;
    //  First modified GRU (backward)
    const flappie_matrix gruB1_iW;
    const flappie_matrix gruB1_sW;
    const flappie_matrix gruB1_b;
    //  Second modified GRU (forward)
    const flappie_matrix gruF2_iW;
    const flappie_matrix gruF2_sW;
    const flappie_matrix gruF2_b;
    //  Third modified GRU (backward)
    const flappie_matrix gruB3_iW;
    const flappie_matrix gruB3_sW;
    const flappie_matrix gruB3_b;
    //  Fourth modified GRU (forward)
    const flappie_matrix gruF4_iW;
    const flappie_matrix gruF4_sW;
    const flappie_matrix gruF4_b;
    //  Fifth modified GRU (backward)
    const flappie_matrix gruB5_iW;
    const flappie_matrix gruB5_sW;
    const flappie_matrix gruB5_b;
    //  Output
    const flappie_matrix FF_W;
    const flappie_matrix FF_b;
} guppy_model;

guppy_model flipflop_r941native_guppy = {
    //  Convolution layer
    .conv_W = &_conv_rnnrf_flipflop_r941native_W,
    .conv_b = &_conv_rnnrf_flipflop_r941native_b,
    .conv_stride = conv_rnnrf_flipflop_r941native_stride,
    //.conv_stride = 2,
    //  First modified GRU (backward)
    .gruB1_iW = &_gruB1_rnnrf_flipflop_r941native_iW,
    .gruB1_sW = &_gruB1_rnnrf_flipflop_r941native_sW,
    .gruB1_b = &_gruB1_rnnrf_flipflop_r941native_b,
    //  Second modified GRU (forward)
    .gruF2_iW = &_gruF2_rnnrf_flipflop_r941native_iW,
    .gruF2_sW = &_gruF2_rnnrf_flipflop_r941native_sW,
    .gruF2_b = &_gruF2_rnnrf_flipflop_r941native_b,
    //  Third modified GRU (backward)
    .gruB3_iW = &_gruB3_rnnrf_flipflop_r941native_iW,
    .gruB3_sW = &_gruB3_rnnrf_flipflop_r941native_sW,
    .gruB3_b = &_gruB3_rnnrf_flipflop_r941native_b,
    //  Fourth modified GRU (forward)
    .gruF4_iW = &_gruF4_rnnrf_flipflop_r941native_iW,
    .gruF4_sW = &_gruF4_rnnrf_flipflop_r941native_sW,
    .gruF4_b = &_gruF4_rnnrf_flipflop_r941native_b,
    //  Fifth modified GRU (backward)
    .gruB5_iW = &_gruB5_rnnrf_flipflop_r941native_iW,
    .gruB5_sW = &_gruB5_rnnrf_flipflop_r941native_sW,
    .gruB5_b = &_gruB5_rnnrf_flipflop_r941native_b,
    //  Output
    .FF_W = &_FF_rnnrf_flipflop_r941native_W,
    .FF_b = &_FF_rnnrf_flipflop_r941native_b
};


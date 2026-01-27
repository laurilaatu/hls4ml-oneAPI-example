#ifndef NNET_INSTR_GEN_H_
#define NNET_INSTR_GEN_H_

namespace nnet {

// hls4ml insert code

template<typename input_t, typename output_t>
void q_dense_1_iq(input_t &inp, output_t &out) {
    

    out[0] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[0]);
    out[1] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[1]);
    out[2] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[2]);
    out[3] = 0;
    out[4] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[4]);
    out[5] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[5]);
    out[6] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[6]);
    out[7] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[7]);
    out[8] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[8]);
    out[9] = 0;
    out[10] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[10]);
    out[11] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[11]);
    out[12] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[12]);
    out[13] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[13]);
    out[14] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[14]);
    out[15] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[15]);
    out[16] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[16]);
    out[17] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[17]);
    out[18] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[18]);
    out[19] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[19]);
    out[20] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[20]);
    out[21] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[21]);
    out[22] = 0;
    out[23] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[23]);
    out[24] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[24]);
    out[25] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[25]);
    out[26] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[26]);
    out[27] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[27]);
    out[28] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[28]);
    out[29] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[29]);
    out[30] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[30]);
    out[31] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[31]);
    out[32] = 0;
    out[33] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[33]);
    out[34] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[34]);
    out[35] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[35]);
    out[36] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[36]);
    out[37] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[37]);
    out[38] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[38]);
    out[39] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[39]);
    out[40] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[40]);
    out[41] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[41]);
    out[42] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[42]);
    out[43] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[43]);
    out[44] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[44]);
    out[45] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[45]);
    out[46] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[46]);
    out[47] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[47]);
    out[48] = 0;
    out[49] = 0;
    out[50] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[50]);
    out[51] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[51]);
    out[52] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[52]);
    out[53] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[53]);
    out[54] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[54]);
    out[55] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[55]);
    out[56] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[56]);
    out[57] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[57]);
    out[58] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[58]);
    out[59] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[59]);
    out[60] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[60]);
    out[61] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[61]);
    out[62] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[62]);
    out[63] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[63]);
}

template<typename input_t, typename output_t>
void q_dense_2_iq(input_t &inp, output_t &out) {
    

    out[0] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[0]);
    out[1] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[1]);
    out[2] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[2]);
    out[3] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[3]);
    out[4] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[4]);
    out[5] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[5]);
    out[6] = 0;
    out[7] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[7]);
    out[8] = 0;
    out[9] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[9]);
    out[10] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[10]);
    out[11] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[11]);
    out[12] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[12]);
    out[13] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[13]);
    out[14] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[14]);
    out[15] = 0;
    out[16] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[16]);
    out[17] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[17]);
    out[18] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[18]);
    out[19] = 0;
    out[20] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[20]);
    out[21] = 0;
    out[22] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[22]);
    out[23] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[23]);
    out[24] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[24]);
    out[25] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[25]);
    out[26] = 0;
    out[27] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[27]);
    out[28] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[28]);
    out[29] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[29]);
    out[30] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[30]);
    out[31] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[31]);
    out[32] = 0;
    out[33] = 0;
    out[34] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[34]);
    out[35] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[35]);
    out[36] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[36]);
    out[37] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[37]);
    out[38] = 0;
    out[39] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[39]);
    out[40] = 0;
    out[41] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[41]);
    out[42] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[42]);
    out[43] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[43]);
    out[44] = 0;
    out[45] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[45]);
    out[46] = 0;
    out[47] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[47]);
    out[48] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[48]);
    out[49] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[49]);
    out[50] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[50]);
    out[51] = 0;
    out[52] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[52]);
    out[53] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[53]);
    out[54] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[54]);
    out[55] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[55]);
    out[56] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[56]);
    out[57] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[57]);
    out[58] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[58]);
    out[59] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[59]);
    out[60] = 0;
    out[61] = 0;
    out[62] = 0;
    out[63] = 0;
}

template<typename input_t, typename output_t>
void q_dense_3_iq(input_t &inp, output_t &out) {
    

    out[0] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[0]);
    out[1] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[1]);
    out[2] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[2]);
    out[3] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[3]);
    out[4] = 0;
    out[5] = 0;
    out[6] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[6]);
    out[7] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[7]);
    out[8] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[8]);
    out[9] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[9]);
    out[10] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[10]);
    out[11] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[11]);
    out[12] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[12]);
    out[13] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[13]);
    out[14] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[14]);
    out[15] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[15]);
    out[16] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[16]);
    out[17] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[17]);
    out[18] = 0;
    out[19] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[19]);
    out[20] = 0;
    out[21] = 0;
    out[22] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[22]);
    out[23] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[23]);
    out[24] = 0;
    out[25] = 0;
    out[26] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[26]);
    out[27] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[27]);
    out[28] = 0;
    out[29] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[29]);
    out[30] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[30]);
    out[31] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[31]);
    out[32] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[32]);
    out[33] = 0;
    out[34] = 0;
    out[35] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[35]);
    out[36] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[36]);
    out[37] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[37]);
    out[38] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[38]);
    out[39] = 0;
    out[40] = 0;
    out[41] = 0;
    out[42] = 0;
    out[43] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[43]);
    out[44] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[44]);
    out[45] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[45]);
    out[46] = 0;
    out[47] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[47]);
    out[48] = ac_fixed<2,-2,false,AC_TRN,AC_WRAP>(inp[48]);
    out[49] = 0;
    out[50] = 0;
    out[51] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[51]);
    out[52] = ac_fixed<3,0,false,AC_TRN,AC_WRAP>(inp[52]);
    out[53] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[53]);
    out[54] = 0;
    out[55] = 0;
    out[56] = 0;
    out[57] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[57]);
    out[58] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[58]);
    out[59] = 0;
    out[60] = ac_fixed<2,-1,false,AC_TRN,AC_WRAP>(inp[60]);
    out[61] = 0;
    out[62] = 0;
    out[63] = 0;
}

} // namespace nnet

#endif

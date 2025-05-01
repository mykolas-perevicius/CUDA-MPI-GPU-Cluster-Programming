// final_project/v4_mpi_cuda/src/alexnet_mpi_cuda.cu
#include <cuda_runtime.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cinttypes>
#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

#include "../include/alexnet.hpp" // Includes LayerParams, dim helpers
#include "../include/layers.hpp"  // Includes CUDA kernel launchers

#define CUDA_CHECK(call)                                    \
  do {                                                      \
    cudaError_t err = call;                                 \
    if(err != cudaSuccess) {                                \
      int rank_for_error; MPI_Comm_rank(MPI_COMM_WORLD, &rank_for_error); \
      fprintf(stderr, "[Rank %d] CUDA error in %s:%d: %s (%d)\n", rank_for_error, __FILE__, __LINE__, cudaGetErrorString(err), err); \
      fflush(stderr); MPI_Abort(MPI_COMM_WORLD, err); } } while(0)

inline int ceil_div(int numerator, int denominator) {
    if (denominator <= 0) { return 0; }
    return (numerator + denominator - 1) / denominator;
}
inline int mapRangeStart(int in_start, int F, int P, int S) {
    if (S <= 0) return 0; int numerator = in_start - F + 1 + P; return ceil_div(numerator, S);
}
inline int mapRangeEnd(int in_end, int /*F*/, int P, int S) {
    if (S <= 0) return -1; int numerator = in_end + P;
    if (numerator >= 0) { return numerator / S; }
    else { return (numerator / S) - ((numerator % S != 0) ? 1 : 0); }
}

void alexnetForwardPassMPI_CUDA(
    const std::vector<float>& h_localInput, int localH,
    int H, int W, int C, const LayerParams& p1, const LayerParams& p2,
    std::vector<float>& h_localOutput, int rank, int size ) {

    if (localH <= 0 || H <= 0 || W <= 0 || C <= 0) { h_localOutput.clear(); return; }

    int haloRows1 = (p1.F > 1) ? p1.F / 2 : 0; int haloRows2 = (p2.F > 1) ? p2.F / 2 : 0;
    bool hasTopHalo = (rank > 0 && haloRows1 > 0); bool hasBotHalo = (rank < size - 1 && haloRows1 > 0);
    int paddedH1 = localH + (hasTopHalo ? haloRows1 : 0) + (hasBotHalo ? haloRows1 : 0);
    int Hc1 = convOutDim(paddedH1, p1.F, p1.P, p1.S); int Wc1 = convOutDim(W, p1.F, p1.P, p1.S);
    int Hp1 = poolOutDim(Hc1, p1.F_pool, p1.S_pool); int Wp1 = poolOutDim(Wc1, p1.F_pool, p1.S_pool); int C1 = p1.K;

    int myGlobalStartRow = 0;
    if (size > 0 && H > 0) { int baseRows = H / size; int extraRows = H % size; for(int r=0; r<rank; ++r) { myGlobalStartRow += baseRows + (r < extraRows ? 1 : 0); } } else if (H <= 0) { fprintf(stderr, "[Rank %d] Error: H invalid (%d).\n", rank, H); MPI_Abort(MPI_COMM_WORLD, 1); }
    int myGlobalEndRow = myGlobalStartRow + localH - 1;
    if (myGlobalEndRow >= H) { fprintf(stderr, "[Rank %d] Error: Global end row (%d) >= H (%d).\n", rank, myGlobalEndRow, H); MPI_Abort(MPI_COMM_WORLD, 1); }

    int conv1_start_ho = mapRangeStart(myGlobalStartRow, p1.F, p1.P, p1.S); int conv1_end_ho = mapRangeEnd(myGlobalEndRow, p1.F, p1.P, p1.S);
    int pool1_start_ho = mapRangeStart(conv1_start_ho, p1.F_pool, 0, p1.S_pool); int pool1_end_ho = mapRangeEnd(conv1_end_ho, p1.F_pool, 0, p1.S_pool);
    int validHp1 = (pool1_end_ho >= pool1_start_ho) ? (pool1_end_ho - pool1_start_ho + 1) : 0;
    int global_conv1_start_for_row0 = mapRangeStart(0, p1.F, p1.P, p1.S); int global_pool1_start_for_row0 = mapRangeStart(global_conv1_start_for_row0, p1.F_pool, 0, p1.S_pool);
    int trimTop1 = pool1_start_ho - global_pool1_start_for_row0;
    if (trimTop1 < 0) trimTop1 = 0;
    if (trimTop1 >= Hp1) { trimTop1 = (Hp1 > 0 ? Hp1 : 0); validHp1 = 0; } else if (Hp1 > 0) { if (trimTop1 + validHp1 > Hp1) { validHp1 = std::max(0, Hp1 - trimTop1); } } else { validHp1 = 0; trimTop1 = 0; }
    if (validHp1 < 0) validHp1 = 0;
    fprintf(stderr, "[Rank %d] Pool1 Trim: Hp1=%d, global_pool1_range=[%d, %d], validHp1=%d, trimTop1=%d\n", rank, Hp1, pool1_start_ho, pool1_end_ho, validHp1, trimTop1);

    bool hasTopHalo2 = (rank > 0 && haloRows2 > 0 && validHp1 > 0); bool hasBotHalo2 = (rank < size - 1 && haloRows2 > 0 && validHp1 > 0);
    int paddedH2 = validHp1 + (hasTopHalo2 ? haloRows2 : 0) + (hasBotHalo2 ? haloRows2 : 0);
    int Hc2 = convOutDim(paddedH2, p2.F, p2.P, p2.S); int Wc2 = convOutDim(Wp1, p2.F, p2.P, p2.S);
    int Hp2 = poolOutDim(Hc2, p2.F_pool, p2.S_pool); int Wp2 = poolOutDim(Wc2, p2.F_pool, p2.S_pool); int C2 = p2.K;
    //fprintf(stderr, "[Rank %d] Calc2: validHp1=%d, paddedH2=%d, Hc2=%d, Wc2=%d, Hp2=%d, Wp2=%d, C2=%d\n", rank, validHp1, paddedH2, Hc2, Wc2, Hp2, Wp2, C2);

    int global_pool1_start = pool1_start_ho; int global_pool1_end = pool1_end_ho;
    int conv2_start_ho = mapRangeStart(global_pool1_start, p2.F, p2.P, p2.S); int conv2_end_ho = mapRangeEnd(global_pool1_end, p2.F, p2.P, p2.S);
    int pool2_start_ho = mapRangeStart(conv2_start_ho, p2.F_pool, 0, p2.S_pool); int pool2_end_ho = mapRangeEnd(conv2_end_ho, p2.F_pool, 0, p2.S_pool);
    int finalLocalH = (pool2_end_ho >= pool2_start_ho) ? (pool2_end_ho - pool2_start_ho + 1) : 0;
    int global_conv2_start_for_row0 = mapRangeStart(global_pool1_start_for_row0, p2.F, p2.P, p2.S); int global_pool2_start_for_row0 = mapRangeStart(global_conv2_start_for_row0, p2.F_pool, 0, p2.S_pool);
    int trimTop2 = pool2_start_ho - global_pool2_start_for_row0;
    if (trimTop2 < 0) trimTop2 = 0;
    if (trimTop2 >= Hp2) { trimTop2 = (Hp2 > 0 ? Hp2 : 0); finalLocalH = 0; } else if (Hp2 > 0) { if (trimTop2 + finalLocalH > Hp2) { finalLocalH = std::max(0, Hp2 - trimTop2); } } else { trimTop2 = 0; finalLocalH = 0; }
    if (finalLocalH < 0) finalLocalH = 0;
    fprintf(stderr, "[Rank %d] Pool2 Trim: Hp2=%d, global_pool2_range=[%d, %d], finalLocalH=%d, trimTop2=%d\n", rank, Hp2, pool2_start_ho, pool2_end_ho, finalLocalH, trimTop2);

    float *d_input_padded1=nullptr, *d_conv1_out=nullptr, *d_pool1_out=nullptr, *d_input_padded2=nullptr, *d_conv2_out=nullptr, *d_pool2_out=nullptr, *d_lrn2_out=nullptr;
    float *d_weights1=nullptr, *d_biases1=nullptr, *d_weights2=nullptr, *d_biases2=nullptr;
    size_t inputPadded1Size=(size_t)paddedH1*W*C, conv1OutSize=(size_t)Hc1*Wc1*C1, pool1OutSize=(size_t)Hp1*Wp1*C1, inputPadded2Size=(size_t)paddedH2*Wp1*C1, conv2OutSize=(size_t)Hc2*Wc2*C2, pool2OutSize=(size_t)Hp2*Wp2*C2, lrn2OutSize=pool2OutSize;
    size_t w1Size=p1.weights.size(), b1Size=p1.biases.size(), w2Size=p2.weights.size(), b2Size=p2.biases.size();
    if(inputPadded1Size>0) CUDA_CHECK(cudaMalloc(&d_input_padded1, inputPadded1Size*sizeof(float))); if(conv1OutSize>0) CUDA_CHECK(cudaMalloc(&d_conv1_out, conv1OutSize*sizeof(float))); if(pool1OutSize>0) CUDA_CHECK(cudaMalloc(&d_pool1_out, pool1OutSize*sizeof(float))); if(inputPadded2Size>0) CUDA_CHECK(cudaMalloc(&d_input_padded2, inputPadded2Size*sizeof(float))); if(conv2OutSize>0) CUDA_CHECK(cudaMalloc(&d_conv2_out, conv2OutSize*sizeof(float))); if(pool2OutSize>0) CUDA_CHECK(cudaMalloc(&d_pool2_out, pool2OutSize*sizeof(float))); if(lrn2OutSize>0) CUDA_CHECK(cudaMalloc(&d_lrn2_out, lrn2OutSize*sizeof(float)));
    if(w1Size>0) CUDA_CHECK(cudaMalloc(&d_weights1, w1Size*sizeof(float))); if(b1Size>0) CUDA_CHECK(cudaMalloc(&d_biases1, b1Size*sizeof(float))); if(w2Size>0) CUDA_CHECK(cudaMalloc(&d_weights2, w2Size*sizeof(float))); if(b2Size>0) CUDA_CHECK(cudaMalloc(&d_biases2, b2Size*sizeof(float)));

    if(w1Size>0 && !p1.weights.empty() && d_weights1) CUDA_CHECK(cudaMemcpy(d_weights1,p1.weights.data(),w1Size*sizeof(float),cudaMemcpyHostToDevice)); if(b1Size>0 && !p1.biases.empty() && d_biases1) CUDA_CHECK(cudaMemcpy(d_biases1,p1.biases.data(),b1Size*sizeof(float),cudaMemcpyHostToDevice)); if(w2Size>0 && !p2.weights.empty() && d_weights2) CUDA_CHECK(cudaMemcpy(d_weights2,p2.weights.data(),w2Size*sizeof(float),cudaMemcpyHostToDevice)); if(b2Size>0 && !p2.biases.empty() && d_biases2) CUDA_CHECK(cudaMemcpy(d_biases2,p2.biases.data(),b2Size*sizeof(float),cudaMemcpyHostToDevice));
    size_t localInputSizeBytes=h_localInput.size()*sizeof(float); size_t inputRowSizeBytes=(W>0&&C>0)?(size_t)W*C*sizeof(float):0; size_t inputHalo1SizeBytes=(size_t)haloRows1*inputRowSizeBytes; size_t inputOffsetElements=(hasTopHalo?inputHalo1SizeBytes:0)/sizeof(float);
    if(localInputSizeBytes>0 && d_input_padded1){ if(inputOffsetElements*sizeof(float)+localInputSizeBytes > inputPadded1Size*sizeof(float)){fprintf(stderr,"[Rank %d] Error: Initial H->D copy bounds.\n",rank);MPI_Abort(MPI_COMM_WORLD,1);} CUDA_CHECK(cudaMemcpy(d_input_padded1+inputOffsetElements,h_localInput.data(),localInputSizeBytes,cudaMemcpyHostToDevice));}

    if(haloRows1>0 && inputRowSizeBytes>0){
        MPI_Request s_up=MPI_REQUEST_NULL, r_up=MPI_REQUEST_NULL, s_down=MPI_REQUEST_NULL, r_down=MPI_REQUEST_NULL;
        std::vector<float> hs_up(haloRows1*W*C),hr_up(haloRows1*W*C),hs_down(haloRows1*W*C),hr_down(haloRows1*W*C);
        int p_up=rank-1, p_down=rank+1; int t_up=0,t_down=1; int h_count=inputHalo1SizeBytes/sizeof(float); if(h_count<=0) goto skip_halo1;
        //fprintf(stderr,"[Rank %d] Halo1: Exchanging %d elem.\n",rank,h_count);
        if(hasTopHalo){MPI_Irecv(hr_up.data(),h_count,MPI_FLOAT,p_up,t_up,MPI_COMM_WORLD,&r_up);} if(hasBotHalo){MPI_Irecv(hr_down.data(),h_count,MPI_FLOAT,p_down,t_down,MPI_COMM_WORLD,&r_down);}
        if(hasBotHalo && d_input_padded1){ size_t offset=inputOffsetElements+(localInputSizeBytes/sizeof(float))-h_count; if(offset*sizeof(float)>=0 && offset+h_count<=inputPadded1Size){ /*fprintf(stderr,"[R%d] H1 S_UP D->H Off=%" PRIu64 " Cnt=%d\n",rank,(uint64_t)offset,h_count);*/ CUDA_CHECK(cudaMemcpy(hs_up.data(),d_input_padded1+offset,inputHalo1SizeBytes,cudaMemcpyDeviceToHost)); MPI_Isend(hs_up.data(),h_count,MPI_FLOAT,p_down,t_up,MPI_COMM_WORLD,&s_up); }else{fprintf(stderr,"[R%d] Err H1 S_UP Off/Bnds\n",rank); MPI_Abort(MPI_COMM_WORLD,1);}}
        if(hasTopHalo && d_input_padded1){ size_t offset=inputOffsetElements; if(offset*sizeof(float)>=0 && offset+h_count<=inputPadded1Size){ /*fprintf(stderr,"[R%d] H1 S_DN D->H Off=%" PRIu64 " Cnt=%d\n",rank,(uint64_t)offset,h_count);*/ CUDA_CHECK(cudaMemcpy(hs_down.data(),d_input_padded1+offset,inputHalo1SizeBytes,cudaMemcpyDeviceToHost)); MPI_Isend(hs_down.data(),h_count,MPI_FLOAT,p_up,t_down,MPI_COMM_WORLD,&s_down); }else{fprintf(stderr,"[R%d] Err H1 S_DN Off/Bnds\n",rank); MPI_Abort(MPI_COMM_WORLD,1);}}
        MPI_Status status;
        if(hasTopHalo){/*fprintf(stderr,"[R%d] H1 Wait RecvUp...\n",rank);*/ MPI_Wait(&r_up,&status); /*fprintf(stderr,"[R%d] H1 Wait RecvUp DONE.\n",rank);*/ if(d_input_padded1 && inputHalo1SizeBytes>0){if(inputHalo1SizeBytes > inputPadded1Size*sizeof(float)){fprintf(stderr,"[R%d] Err H1 RcvUp Dest Small\n",rank); MPI_Abort(MPI_COMM_WORLD,1);} /*fprintf(stderr,"[R%d] H1 RecvUp H->D Off=0 Cnt=%d\n",rank,h_count);*/ CUDA_CHECK(cudaMemcpy(d_input_padded1,hr_up.data(),inputHalo1SizeBytes,cudaMemcpyHostToDevice));} if(s_down!=MPI_REQUEST_NULL){/*fprintf(stderr,"[R%d] H1 Wait SendDn...\n",rank);*/ MPI_Wait(&s_down,&status); /*fprintf(stderr,"[R%d] H1 Wait SendDn DONE.\n",rank);*/}}
        if(hasBotHalo){/*fprintf(stderr,"[R%d] H1 Wait RecvDn...\n",rank);*/ MPI_Wait(&r_down,&status); /*fprintf(stderr,"[R%d] H1 Wait RecvDn DONE.\n",rank);*/ size_t offset=inputOffsetElements+(localInputSizeBytes/sizeof(float)); if(d_input_padded1 && inputHalo1SizeBytes>0 && offset+h_count<=inputPadded1Size){ /*fprintf(stderr,"[R%d] H1 RecvDn H->D Off=%" PRIu64 " Cnt=%d\n",rank,(uint64_t)offset,h_count);*/ CUDA_CHECK(cudaMemcpy(d_input_padded1+offset,hr_down.data(),inputHalo1SizeBytes,cudaMemcpyHostToDevice));}else if(inputHalo1SizeBytes>0){fprintf(stderr,"[R%d] Err H1 RcvDn Off/Bnds\n",rank);MPI_Abort(MPI_COMM_WORLD,1);} if(s_up!=MPI_REQUEST_NULL){/*fprintf(stderr,"[R%d] H1 Wait SendUp...\n",rank);*/ MPI_Wait(&s_up,&status); /*fprintf(stderr,"[R%d] H1 Wait SendUp DONE.\n",rank);*/}}
    } skip_halo1:;

    if(Hc1>0 && Wc1>0 && C1>0 && d_conv1_out && d_input_padded1 && d_weights1 && d_biases1){ /*fprintf(stderr,"[R%d] Launch Conv1\n",rank);*/ cudaConvLayer(d_conv1_out,d_input_padded1,d_weights1,d_biases1,paddedH1,W,C,p1.K,p1.F,p1.S,p1.P); cudaReluLayer(d_conv1_out,conv1OutSize);}
    if(Hp1>0 && Wp1>0 && C1>0 && d_pool1_out && d_conv1_out){ /*fprintf(stderr,"[R%d] Launch Pool1\n",rank);*/ cudaMaxPoolLayer(d_pool1_out,d_conv1_out,Hc1,Wc1,C1,p1.F_pool,p1.S_pool);}

    size_t pool1RowSizeBytes=(Wp1>0 && C1>0)?(size_t)Wp1*C1*sizeof(float):0;
    if(haloRows2>0 && validHp1>0 && pool1RowSizeBytes>0 && d_pool1_out){
        MPI_Request s_up=MPI_REQUEST_NULL, r_up=MPI_REQUEST_NULL, s_down=MPI_REQUEST_NULL, r_down=MPI_REQUEST_NULL;
        size_t pool1HaloSizeBytes=(size_t)haloRows2*pool1RowSizeBytes; if(haloRows2>0 && pool1RowSizeBytes>SIZE_MAX/(unsigned long)haloRows2){fprintf(stderr,"[R%d] Err H2 Size Ovflw\n",rank); MPI_Abort(MPI_COMM_WORLD,1);} if(pool1HaloSizeBytes==0) goto skip_halo2;
        std::vector<float> hs_up(pool1HaloSizeBytes/sizeof(float)),hr_up(pool1HaloSizeBytes/sizeof(float)),hs_down(pool1HaloSizeBytes/sizeof(float)),hr_down(pool1HaloSizeBytes/sizeof(float));
        int p_up=rank-1, p_down=rank+1; int t_up=2,t_down=3; int h_count=pool1HaloSizeBytes/sizeof(float); if(h_count<=0) goto skip_halo2;
        //fprintf(stderr,"[R%d] Halo2: Exchanging %d elem.\n",rank,h_count);
        size_t offsetBytes=(size_t)trimTop1*pool1RowSizeBytes; if(offsetBytes>pool1OutSize*sizeof(float)){fprintf(stderr,"[R%d] Err H2 Trim Offset\n",rank);MPI_Abort(MPI_COMM_WORLD,1);}
        float* valid_start=d_pool1_out+offsetBytes/sizeof(float); size_t validSizeBytes=(size_t)validHp1*pool1RowSizeBytes; if(offsetBytes+validSizeBytes > pool1OutSize*sizeof(float)){fprintf(stderr,"[R%d] Err H2 Valid Region Bnds\n",rank);MPI_Abort(MPI_COMM_WORLD,1);}

        if(hasTopHalo2){MPI_Irecv(hr_up.data(),h_count,MPI_FLOAT,p_up,t_up,MPI_COMM_WORLD,&r_up);} if(hasBotHalo2){MPI_Irecv(hr_down.data(),h_count,MPI_FLOAT,p_down,t_down,MPI_COMM_WORLD,&r_down);}
        if(hasBotHalo2){size_t offset=validSizeBytes-pool1HaloSizeBytes; if(pool1HaloSizeBytes>validSizeBytes){fprintf(stderr,"[R%d] Err H2 S_UP Size %" PRIu64 ">%" PRIu64 "\n",rank,(uint64_t)pool1HaloSizeBytes,(uint64_t)validSizeBytes); MPI_Abort(MPI_COMM_WORLD,1);} if(offsetBytes+offset+pool1HaloSizeBytes > pool1OutSize*sizeof(float)){fprintf(stderr,"[R%d] Err H2 S_UP Src Bnds\n",rank);MPI_Abort(MPI_COMM_WORLD,1);} /*fprintf(stderr,"[R%d] H2 S_UP D->H Off=%" PRIu64 " Cnt=%d\n",rank,(uint64_t)(offsetBytes/sizeof(float)+offset/sizeof(float)),h_count);*/ CUDA_CHECK(cudaMemcpy(hs_up.data(),valid_start+offset/sizeof(float),pool1HaloSizeBytes,cudaMemcpyDeviceToHost)); MPI_Isend(hs_up.data(),h_count,MPI_FLOAT,p_down,t_up,MPI_COMM_WORLD,&s_up);}
        if(hasTopHalo2){if(pool1HaloSizeBytes>validSizeBytes){fprintf(stderr,"[R%d] Err H2 S_DN Size %" PRIu64 ">%" PRIu64 "\n",rank,(uint64_t)pool1HaloSizeBytes,(uint64_t)validSizeBytes);MPI_Abort(MPI_COMM_WORLD,1);} if(offsetBytes+pool1HaloSizeBytes > pool1OutSize*sizeof(float)){fprintf(stderr,"[R%d] Err H2 S_DN Src Bnds\n",rank);MPI_Abort(MPI_COMM_WORLD,1);} /*fprintf(stderr,"[R%d] H2 S_DN D->H Off=%" PRIu64 " Cnt=%d\n",rank,(uint64_t)(offsetBytes/sizeof(float)),h_count);*/ CUDA_CHECK(cudaMemcpy(hs_down.data(),valid_start,pool1HaloSizeBytes,cudaMemcpyDeviceToHost)); MPI_Isend(hs_down.data(),h_count,MPI_FLOAT,p_up,t_down,MPI_COMM_WORLD,&s_down);}

        size_t currentOffsetBytes=0;
        if(hasTopHalo2){MPI_Wait(&r_up,MPI_STATUS_IGNORE); if(d_input_padded2 && pool1HaloSizeBytes>0 && pool1HaloSizeBytes<=inputPadded2Size*sizeof(float)){/*fprintf(stderr,"[R%d] H2 RcvUp H->D Off=0 Cnt=%d\n",rank,h_count);*/ CUDA_CHECK(cudaMemcpy(d_input_padded2,hr_up.data(),pool1HaloSizeBytes,cudaMemcpyHostToDevice));}else if(pool1HaloSizeBytes>0){fprintf(stderr,"[R%d] Err H2 RcvUp Dest Bnds/Null\n",rank);MPI_Abort(MPI_COMM_WORLD,1);} currentOffsetBytes+=pool1HaloSizeBytes; if(s_down!=MPI_REQUEST_NULL){MPI_Wait(&s_down,MPI_STATUS_IGNORE);}}
        if(validHp1>0 && validSizeBytes>0 && d_input_padded2){if(currentOffsetBytes+validSizeBytes > inputPadded2Size*sizeof(float)){fprintf(stderr,"[R%d] Err H2 D->D Dest Bnds\n",rank);MPI_Abort(MPI_COMM_WORLD,1);} /*fprintf(stderr,"[R%d] H2 Copy D->D DestOff=%" PRIu64 " SrcOff=%" PRIu64 " Cnt=%" PRIu64 "\n",rank,(uint64_t)(currentOffsetBytes/sizeof(float)),(uint64_t)(offsetBytes/sizeof(float)),(uint64_t)(validSizeBytes/sizeof(float)));*/ CUDA_CHECK(cudaMemcpy(d_input_padded2+currentOffsetBytes/sizeof(float),valid_start,validSizeBytes,cudaMemcpyDeviceToDevice)); currentOffsetBytes+=validSizeBytes;}
        if(hasBotHalo2){MPI_Wait(&r_down,MPI_STATUS_IGNORE); if(d_input_padded2 && pool1HaloSizeBytes>0 && currentOffsetBytes+pool1HaloSizeBytes <= inputPadded2Size*sizeof(float)){/*fprintf(stderr,"[R%d] H2 RcvDn H->D Off=%" PRIu64 " Cnt=%d\n",rank,(uint64_t)(currentOffsetBytes/sizeof(float)),h_count);*/ CUDA_CHECK(cudaMemcpy(d_input_padded2+currentOffsetBytes/sizeof(float),hr_down.data(),pool1HaloSizeBytes,cudaMemcpyHostToDevice));}else if(pool1HaloSizeBytes>0){fprintf(stderr,"[R%d] Err H2 RcvDn Dest Bnds/Null\n",rank);MPI_Abort(MPI_COMM_WORLD,1);} if(s_up!=MPI_REQUEST_NULL){MPI_Wait(&s_up,MPI_STATUS_IGNORE);}}

    }else if(validHp1>0 && pool1RowSizeBytes>0 && d_pool1_out && d_input_padded2){
        size_t offsetBytes=(size_t)trimTop1*pool1RowSizeBytes; float* valid_start=d_pool1_out+offsetBytes/sizeof(float); size_t validSizeBytes=(size_t)validHp1*pool1RowSizeBytes;
        if(offsetBytes+validSizeBytes > pool1OutSize*sizeof(float)){fprintf(stderr,"[R%d] Err H2 D->D Src Bnds (NoHalo)\n",rank);MPI_Abort(MPI_COMM_WORLD,1);} if(validSizeBytes > inputPadded2Size*sizeof(float)){fprintf(stderr,"[R%d] Err H2 D->D Dest Bnds (NoHalo)\n",rank);MPI_Abort(MPI_COMM_WORLD,1);}
        if(validSizeBytes>0){/*fprintf(stderr,"[R%d] H2 Copy D->D (NoHalo) DestOff=0 SrcOff=%" PRIu64 " Cnt=%" PRIu64 "\n",rank,(uint64_t)(offsetBytes/sizeof(float)),(uint64_t)(validSizeBytes/sizeof(float)));*/ CUDA_CHECK(cudaMemcpy(d_input_padded2,valid_start,validSizeBytes,cudaMemcpyDeviceToDevice));}
    }else if(d_input_padded2 && inputPadded2Size>0){/*fprintf(stderr,"[R%d] H2 Zeroing\n",rank);*/ CUDA_CHECK(cudaMemset(d_input_padded2,0,inputPadded2Size*sizeof(float)));}
    skip_halo2:;

    if(Hc2>0 && Wc2>0 && C2>0 && d_conv2_out && d_input_padded2 && d_weights2 && d_biases2){/*fprintf(stderr,"[R%d] Launch Conv2\n",rank);*/ cudaConvLayer(d_conv2_out,d_input_padded2,d_weights2,d_biases2,paddedH2,Wp1,C1,p2.K,p2.F,p2.S,p2.P); cudaReluLayer(d_conv2_out,conv2OutSize);}
    if(Hp2>0 && Wp2>0 && C2>0 && d_pool2_out && d_conv2_out){/*fprintf(stderr,"[R%d] Launch Pool2\n",rank);*/ cudaMaxPoolLayer(d_pool2_out,d_conv2_out,Hc2,Wc2,C2,p2.F_pool,p2.S_pool);}
    if(Hp2>0 && Wp2>0 && C2>0 && d_lrn2_out && d_pool2_out){/*fprintf(stderr,"[R%d] Launch LRN2\n",rank);*/ cudaLRNLayer(d_lrn2_out,d_pool2_out,Hp2,Wp2,C2,p2.N_lrn,p2.alpha,p2.beta,p2.k_lrn);}

    size_t finalRowSizeBytes=(Wp2>0 && C2>0)?(size_t)Wp2*C2*sizeof(float):0; size_t finalLocalSizeBytes=(size_t)finalLocalH*finalRowSizeBytes;
    if(finalLocalSizeBytes>0 && finalLocalH>0){if((unsigned long)finalLocalH > SIZE_MAX/((unsigned long)Wp2*C2)){fprintf(stderr,"[R%d] Err Final Size Ovflw\n",rank);MPI_Abort(MPI_COMM_WORLD,1);} h_localOutput.resize(finalLocalSizeBytes/sizeof(float));} else {h_localOutput.clear(); finalLocalSizeBytes=0;}
    if(finalLocalH>0 && finalLocalSizeBytes>0 && d_lrn2_out){
        size_t offsetBytes=(size_t)trimTop2*finalRowSizeBytes;
        if(offsetBytes+finalLocalSizeBytes > lrn2OutSize*sizeof(float)){fprintf(stderr,"[R%d] Err Final D->H Src Bnds Off=%" PRIu64 " Sz=%" PRIu64 " BufSz=%" PRIu64 " finalH=%d trimT2=%d Hp2=%d Wp2=%d C2=%d\n",rank,(uint64_t)offsetBytes,(uint64_t)finalLocalSizeBytes,(uint64_t)(lrn2OutSize*sizeof(float)),finalLocalH,trimTop2,Hp2,Wp2,C2); MPI_Abort(MPI_COMM_WORLD,1);}
        //fprintf(stderr,"[R%d] Final D->H DestSz=%" PRIu64 " SrcOff=%" PRIu64 " CpySz=%" PRIu64 "\n",rank,(uint64_t)h_localOutput.size(),(uint64_t)(offsetBytes/sizeof(float)),(uint64_t)(finalLocalSizeBytes/sizeof(float)));
        CUDA_CHECK(cudaMemcpy(h_localOutput.data(),d_lrn2_out+offsetBytes/sizeof(float),finalLocalSizeBytes,cudaMemcpyDeviceToHost));
    }

    if(d_input_padded1) cudaFree(d_input_padded1); if(d_conv1_out) cudaFree(d_conv1_out); if(d_pool1_out) cudaFree(d_pool1_out); if(d_input_padded2) cudaFree(d_input_padded2); if(d_conv2_out) cudaFree(d_conv2_out); if(d_pool2_out) cudaFree(d_pool2_out); if(d_lrn2_out) cudaFree(d_lrn2_out);
    if(d_weights1) cudaFree(d_weights1); if(d_biases1) cudaFree(d_biases1); if(d_weights2) cudaFree(d_weights2); if(d_biases2) cudaFree(d_biases2);
    //fprintf(stderr, "[Rank %d] alexnetForwardPassMPI_CUDA finished.\n", rank);
}

// === append to the end of final_project/v4_mpi_cuda/src/alexnet_mpi_cuda.cu ===
void alexnetTileForwardCUDA(const float* d_input,
                            const LayerParams& p1,const LayerParams& p2,
                            int H,int W,int C,float* d_output)
{
    int Hc1=convOutDim(H ,p1.F,p1.P,p1.S);
    int Wc1=convOutDim(W ,p1.F,p1.P,p1.S);
    int Hp1=poolOutDim(Hc1,p1.F_pool,p1.S_pool);
    int Wp1=poolOutDim(Wc1,p1.F_pool,p1.S_pool);
    int C1 =p1.K;

    int Hc2=convOutDim(Hp1,p2.F,p2.P,p2.S);
    int Wc2=convOutDim(Wp1,p2.F,p2.P,p2.S);
    int Hp2=poolOutDim(Hc2,p2.F_pool,p2.S_pool);
    int Wp2=poolOutDim(Wc2,p2.F_pool,p2.S_pool);
    int C2 =p2.K;

    size_t nConv1=(size_t)Hc1*Wc1*C1, nPool1=(size_t)Hp1*Wp1*C1,
           nConv2=(size_t)Hc2*Wc2*C2, nPool2=(size_t)Hp2*Wp2*C2;

    float *d_c1,*d_p1,*d_c2,*d_p2;
    CUDA_CHECK(cudaMalloc(&d_c1,nConv1*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p1,nPool1*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c2,nConv2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p2,nPool2*sizeof(float)));

    float *dw1,*db1,*dw2,*db2;
    CUDA_CHECK(cudaMalloc(&dw1,p1.weights.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db1,p1.biases .size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw2,p2.weights.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db2,p2.biases .size()*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dw1,p1.weights.data(),p1.weights.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db1,p1.biases .data(),p1.biases .size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw2,p2.weights.data(),p2.weights.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db2,p2.biases .data(),p2.biases .size()*sizeof(float),cudaMemcpyHostToDevice));

    cudaConvLayer(d_c1,d_input ,dw1,db1,H ,W ,C ,p1.K,p1.F,p1.S,p1.P);
    cudaReluLayer(d_c1,nConv1);
    cudaMaxPoolLayer(d_p1,d_c1,Hc1,Wc1,C1,p1.F_pool,p1.S_pool);

    cudaConvLayer(d_c2,d_p1,dw2,db2,Hp1,Wp1,C1,p2.K,p2.F,p2.S,p2.P);
    cudaReluLayer(d_c2,nConv2);
    cudaMaxPoolLayer(d_p2,d_c2,Hc2,Wc2,C2,p2.F_pool,p2.S_pool);

    cudaLRNLayer(d_output,d_p2,Hp2,Wp2,C2,p2.N_lrn,p2.alpha,p2.beta,p2.k_lrn);

    cudaFree(d_c1); cudaFree(d_p1); cudaFree(d_c2); cudaFree(d_p2);
    cudaFree(dw1);  cudaFree(db1);  cudaFree(dw2);  cudaFree(db2);
}

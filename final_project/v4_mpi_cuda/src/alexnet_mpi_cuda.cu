// final_project/v4_mpi_cuda/src/alexnet_mpi_cuda.cu
#include <cuda_runtime.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream> // For debug prints
#include <cstdio>   // For fprintf, stderr, PRIu64 with cinttypes
#include <cstdlib>  // For exit
#include <climits>  // Include for SIZE_MAX and INT_MAX
#include <cinttypes> // Include for PRIu64 macro
#ifndef SIZE_MAX // Define SIZE_MAX if not defined (e.g., by older standards/compilers)
#define SIZE_MAX ((size_t)-1)
#endif


#include "../include/alexnet.hpp" // Includes LayerParams, dim helpers
#include "../include/layers.hpp"  // Includes CUDA kernel launchers

// Helper macro for checking CUDA errors from .cu files
#define CUDA_CHECK(call)                                    \
  do {                                                      \
    cudaError_t err = call;                                 \
    if(err != cudaSuccess) {                                \
      int rank_for_error; MPI_Comm_rank(MPI_COMM_WORLD, &rank_for_error); \
      fprintf(stderr,                                       \
        "[Rank %d] CUDA error in %s:%d: %s (%d)\n",         \
         rank_for_error, __FILE__, __LINE__, cudaGetErrorString(err), err);  \
      fflush(stderr); /* Ensure message prints before abort */ \
      MPI_Abort(MPI_COMM_WORLD, err); /* Coordinated abort */ \
    }                                                       \
  } while(0)

// Integer ceiling division helper
inline int ceil_div(int numerator, int denominator) {
    if (denominator == 0) return 0; // Or handle error appropriately
    // Ensure calculation handles negative numerator correctly if needed, though unlikely here.
    return (numerator + denominator - 1) / denominator;
}

// Helper to calculate the start index of the output range based on input range start
inline int mapRangeStart(int in_start, int F, int P, int S) {
    if (S <= 0) return 0;
    // out_start = ceil((in_start - F + 1 + P) / S)
    // Need to handle potential negative intermediate value correctly for ceil
    int numerator = in_start - F + 1 + P;
    // Manual ceil for potentially negative numerator: floor((num + den - 1) / den) works for positive den
    // Simpler for positive S: (numerator > 0) ? (numerator + S - 1) / S : numerator / S;
    // Even simpler: use the helper
     return ceil_div(numerator, S);
}

// Helper to calculate the end index of the output range based on input range end
inline int mapRangeEnd(int in_end, int /*F*/, int P, int S) {
     if (S <= 0) return -1; // Indicate invalid range
    // out_end = floor((in_end + P) / S)
    int numerator = in_end + P;
    if (numerator < 0) {
        // Handle negative numerator for floor division if necessary, though unlikely here.
        // Standard integer division truncates towards zero. For floor with negative: (num - den + 1) / den ?
        // Let's assume in_end+P is usually non-negative.
        return (numerator / S) - ( (numerator % S != 0 && numerator < 0) ? 1 : 0); // More robust floor
    } else {
        return numerator / S; // Standard integer division works as floor for non-negative
    }
}


// --- alexnetForwardPassMPI_CUDA Implementation ---
void alexnetForwardPassMPI_CUDA(
    const std::vector<float>& h_localInput, // Local input data slice on host
    int localH,          // Height of the local input slice
    int H, int W, int C, // *** USE PASSED H *** Original Dimensions (H, W, C)
    const LayerParams& p1, // Params for Block 1 (Conv1, Pool1)
    const LayerParams& p2, // Params for Block 2 (Conv2, Pool2, LRN2)
    std::vector<float>& h_localOutput, // Local output slice on host (resized by function)
    int rank, int size   // MPI rank and size for halo logic
) {

    // --- 0. Handle Empty Input Case ---
    if (localH <= 0 || H <= 0 || W <= 0 || C <= 0) {
        h_localOutput.clear();
         // if(rank == 0) fprintf(stderr, "[Rank %d] Info: Skipping computation due to zero input/dimension (localH=%d, H=%d, W=%d, C=%d).\n", rank, localH, H, W, C);
        return;
    }

    // --- 1. Calculate Dimensions & Halo Sizes ---
    int haloRows1 = (p1.F > 1) ? p1.F / 2 : 0;
    int haloRows2 = (p2.F > 1) ? p2.F / 2 : 0;

    bool hasTopHalo = (rank > 0 && haloRows1 > 0);
    bool hasBotHalo = (rank < size - 1 && haloRows1 > 0);
    int paddedH1 = localH + (hasTopHalo ? haloRows1 : 0) + (hasBotHalo ? haloRows1 : 0);

    // Calculate overall dimensions of intermediate layers based on *padded* height
    // These represent the buffer sizes needed.
    int Hc1 = convOutDim(paddedH1, p1.F, p1.P, p1.S);
    int Wc1 = convOutDim(W, p1.F, p1.P, p1.S);
    int Hp1 = poolOutDim(Hc1, p1.F_pool, p1.S_pool);
    int Wp1 = poolOutDim(Wc1, p1.F_pool, p1.S_pool);
    int C1 = p1.K;
    // *** DEBUG PRINT ***
    // fprintf(stderr, "[Rank %d] Calc1: localH=%d, paddedH1=%d, Hc1=%d, Wc1=%d, Hp1=%d, Wp1=%d, C1=%d\n", rank, localH, paddedH1, Hc1, Wc1, Hp1, Wp1, C1);


    // --- Calculate Trimming for Pool1 ---
    // Find the global row indices for this rank's original input data
    int myGlobalStartRow = 0;
    if (size > 0 && H > 0) {
         int baseRows = H / size; int extraRows = H % size;
         for(int r=0; r<rank; ++r) { myGlobalStartRow += baseRows + (r < extraRows ? 1 : 0); }
    } else if (H <= 0) { fprintf(stderr, "[Rank %d] Error: Total height H is invalid (%d).\n", rank, H); MPI_Abort(MPI_COMM_WORLD, 1); }
    int myGlobalEndRow = myGlobalStartRow + localH - 1;
    if (myGlobalEndRow >= H) { fprintf(stderr, "[Rank %d] Error: Calculated global end row (%d) >= total height (%d). localH=%d, startRow=%d\n", rank, myGlobalEndRow, H, localH, myGlobalStartRow); MPI_Abort(MPI_COMM_WORLD, 1); }
    // *** DEBUG PRINT ***
    // fprintf(stderr, "[Rank %d] Global Rows: Start=%d, End=%d (localH=%d)\n", rank, myGlobalStartRow, myGlobalEndRow, localH);


    // Map these global start/end rows through the layers using CORRECT range mapping
    // Map Conv1
    int conv1_start_ho = mapRangeStart(myGlobalStartRow, p1.F, p1.P, p1.S);
    int conv1_end_ho   = mapRangeEnd(myGlobalEndRow, p1.F, p1.P, p1.S);
    // Map Pool1 (using Conv1's output range as input)
    int pool1_start_ho = mapRangeStart(conv1_start_ho, p1.F_pool, 0, p1.S_pool); // Pool P=0
    int pool1_end_ho   = mapRangeEnd(conv1_end_ho, p1.F_pool, 0, p1.S_pool); // Pool P=0

    // Calculate valid height for Pool1 output on this rank
    int validHp1 = (pool1_end_ho >= pool1_start_ho) ? (pool1_end_ho - pool1_start_ho + 1) : 0;

    // Calculate the *local* start index (trimTop1) within the Hp1-sized buffer
    // Need the global start index if rank 0 processed input row 0
    int global_conv1_start_for_row0 = mapRangeStart(0, p1.F, p1.P, p1.S);
    int global_pool1_start_for_row0 = mapRangeStart(global_conv1_start_for_row0, p1.F_pool, 0, p1.S_pool);

    int trimTop1 = pool1_start_ho - global_pool1_start_for_row0;

    // Clamp trimTop1 and validHp1 based on the actual buffer size Hp1
    if (trimTop1 < 0) trimTop1 = 0;
    if (trimTop1 >= Hp1) { // If start offset is beyond buffer size
        trimTop1 = (Hp1 > 0 ? Hp1 : 0);
        validHp1 = 0;
    } else { // If start offset is within buffer, check end
        if (trimTop1 + validHp1 > Hp1) { // If calculated valid region extends beyond buffer
            validHp1 = std::max(0, Hp1 - trimTop1); // Clamp valid height
        }
    }
    if (validHp1 < 0) validHp1 = 0; // Final safety
    // *** DEBUG PRINT ***
    fprintf(stderr, "[Rank %d] Pool1 Trim: Hp1=%d, global_pool1_range=[%d, %d], validHp1=%d, trimTop1=%d\n", rank, Hp1, pool1_start_ho, pool1_end_ho, validHp1, trimTop1);


    // --- Dimensions for Block 2 ---
    bool hasTopHalo2 = (rank > 0 && haloRows2 > 0 && validHp1 > 0);
    bool hasBotHalo2 = (rank < size - 1 && haloRows2 > 0 && validHp1 > 0);
    int paddedH2 = validHp1 + (hasTopHalo2 ? haloRows2 : 0) + (hasBotHalo2 ? haloRows2 : 0);

    // Overall dimensions based on padded height
    int Hc2 = convOutDim(paddedH2, p2.F, p2.P, p2.S);
    int Wc2 = convOutDim(Wp1, p2.F, p2.P, p2.S);
    int Hp2 = poolOutDim(Hc2, p2.F_pool, p2.S_pool);
    int Wp2 = poolOutDim(Wc2, p2.F_pool, p2.S_pool);
    int C2 = p2.K;
    // *** DEBUG PRINT ***
    // fprintf(stderr, "[Rank %d] Calc2: validHp1=%d, paddedH2=%d, Hc2=%d, Wc2=%d, Hp2=%d, Wp2=%d, C2=%d\n", rank, validHp1, paddedH2, Hc2, Wc2, Hp2, Wp2, C2);


    // --- Calculate Trimming for Pool2 ---
    // Map the global Pool1 output range through Conv2 and Pool2
    int global_pool1_start = pool1_start_ho; // Global start index of this rank's valid Pool1 output
    int global_pool1_end = pool1_end_ho;     // Global end index

    // Map Conv2
    int conv2_start_ho = mapRangeStart(global_pool1_start, p2.F, p2.P, p2.S);
    int conv2_end_ho   = mapRangeEnd(global_pool1_end, p2.F, p2.P, p2.S);
    // Map Pool2
    int pool2_start_ho = mapRangeStart(conv2_start_ho, p2.F_pool, 0, p2.S_pool);
    int pool2_end_ho   = mapRangeEnd(conv2_end_ho, p2.F_pool, 0, p2.S_pool);

    // Calculate final valid height for this rank
    int finalLocalH = (pool2_end_ho >= pool2_start_ho) ? (pool2_end_ho - pool2_start_ho + 1) : 0;

    // Calculate trimTop2 relative to the Hp2-sized buffer
    // Need the global start index if rank 0 processed input pool1 row 0 (derived from input row 0)
    int global_conv2_start_for_row0 = mapRangeStart(global_pool1_start_for_row0, p2.F, p2.P, p2.S);
    int global_pool2_start_for_row0 = mapRangeStart(global_conv2_start_for_row0, p2.F_pool, 0, p2.S_pool);

    int trimTop2 = pool2_start_ho - global_pool2_start_for_row0;

    // Clamp trimTop2 and finalLocalH based on actual buffer size Hp2
     if (trimTop2 < 0) trimTop2 = 0;
     if (trimTop2 >= Hp2) {
         trimTop2 = (Hp2 > 0 ? Hp2 : 0);
         finalLocalH = 0;
     } else {
         if (trimTop2 + finalLocalH > Hp2) {
             finalLocalH = std::max(0, Hp2 - trimTop2);
         }
     }
     if (finalLocalH < 0) finalLocalH = 0; // Final safety
     // *** DEBUG PRINT ***
     fprintf(stderr, "[Rank %d] Pool2 Trim: Hp2=%d, global_pool2_range=[%d, %d], finalLocalH=%d, trimTop2=%d\n", rank, Hp2, pool2_start_ho, pool2_end_ho, finalLocalH, trimTop2);


    // --- 2. Allocate Device Memory ---
    // (Allocation code remains the same)
    float *d_input_padded1 = nullptr, *d_conv1_out = nullptr, *d_pool1_out = nullptr,
          *d_input_padded2 = nullptr, *d_conv2_out = nullptr, *d_pool2_out = nullptr,
          *d_lrn2_out = nullptr;
    float *d_weights1 = nullptr, *d_biases1 = nullptr,
          *d_weights2 = nullptr, *d_biases2 = nullptr;

    size_t inputPadded1Size = (size_t)paddedH1 * W * C;
    size_t conv1OutSize = (size_t)Hc1 * Wc1 * C1;
    size_t pool1OutSize = (size_t)Hp1 * Wp1 * C1;
    size_t inputPadded2Size = (size_t)paddedH2 * Wp1 * C1;
    size_t conv2OutSize = (size_t)Hc2 * Wc2 * C2;
    size_t pool2OutSize = (size_t)Hp2 * Wp2 * C2;
    size_t lrn2OutSize = pool2OutSize;

    size_t w1Size = p1.weights.size(); size_t b1Size = p1.biases.size();
    size_t w2Size = p2.weights.size(); size_t b2Size = p2.biases.size();

    if (inputPadded1Size > 0) CUDA_CHECK(cudaMalloc(&d_input_padded1, inputPadded1Size * sizeof(float))); else d_input_padded1=nullptr;
    if (conv1OutSize > 0)     CUDA_CHECK(cudaMalloc(&d_conv1_out, conv1OutSize * sizeof(float)));     else d_conv1_out = nullptr;
    if (pool1OutSize > 0)     CUDA_CHECK(cudaMalloc(&d_pool1_out, pool1OutSize * sizeof(float)));     else d_pool1_out = nullptr;
    if (inputPadded2Size > 0) CUDA_CHECK(cudaMalloc(&d_input_padded2, inputPadded2Size * sizeof(float))); else d_input_padded2 = nullptr;
    if (conv2OutSize > 0)     CUDA_CHECK(cudaMalloc(&d_conv2_out, conv2OutSize * sizeof(float)));     else d_conv2_out = nullptr;
    if (pool2OutSize > 0)     CUDA_CHECK(cudaMalloc(&d_pool2_out, pool2OutSize * sizeof(float)));     else d_pool2_out = nullptr;
    if (lrn2OutSize > 0)      CUDA_CHECK(cudaMalloc(&d_lrn2_out, lrn2OutSize * sizeof(float)));      else d_lrn2_out = nullptr;

    if (w1Size > 0) CUDA_CHECK(cudaMalloc(&d_weights1, w1Size * sizeof(float))); else d_weights1 = nullptr;
    if (b1Size > 0) CUDA_CHECK(cudaMalloc(&d_biases1, b1Size * sizeof(float))); else d_biases1 = nullptr;
    if (w2Size > 0) CUDA_CHECK(cudaMalloc(&d_weights2, w2Size * sizeof(float))); else d_weights2 = nullptr;
    if (b2Size > 0) CUDA_CHECK(cudaMalloc(&d_biases2, b2Size * sizeof(float))); else d_biases2 = nullptr;


    // --- 3. Copy Initial Data Host -> Device ---
    // (Copy code remains the same, including bounds checks)
    if (w1Size > 0 && !p1.weights.empty() && d_weights1) CUDA_CHECK(cudaMemcpy(d_weights1, p1.weights.data(), w1Size * sizeof(float), cudaMemcpyHostToDevice));
    if (b1Size > 0 && !p1.biases.empty() && d_biases1)   CUDA_CHECK(cudaMemcpy(d_biases1, p1.biases.data(), b1Size * sizeof(float), cudaMemcpyHostToDevice));
    if (w2Size > 0 && !p2.weights.empty() && d_weights2) CUDA_CHECK(cudaMemcpy(d_weights2, p2.weights.data(), w2Size * sizeof(float), cudaMemcpyHostToDevice));
    if (b2Size > 0 && !p2.biases.empty() && d_biases2)   CUDA_CHECK(cudaMemcpy(d_biases2, p2.biases.data(), b2Size * sizeof(float), cudaMemcpyHostToDevice));

    size_t localInputSizeBytes = h_localInput.size() * sizeof(float);
    size_t inputRowSizeBytes = (W > 0 && C > 0) ? (size_t)W * C * sizeof(float) : 0;
    size_t inputHalo1SizeBytes = (size_t)haloRows1 * inputRowSizeBytes;
    size_t inputOffsetElements = (hasTopHalo ? inputHalo1SizeBytes : 0) / sizeof(float);

    if (localInputSizeBytes > 0 && d_input_padded1 != nullptr) {
        if (inputOffsetElements * sizeof(float) + localInputSizeBytes > inputPadded1Size * sizeof(float)) {
             fprintf(stderr, "[Rank %d] Error: Initial H->D copy exceeds destination buffer bounds (Offset=%" PRIu64 ", Size=%" PRIu64 ", BufferSize=%" PRIu64 ").\n", rank, (uint64_t)(inputOffsetElements*sizeof(float)), (uint64_t)localInputSizeBytes, (uint64_t)(inputPadded1Size*sizeof(float)) ); MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaMemcpy(d_input_padded1 + inputOffsetElements, h_localInput.data(), localInputSizeBytes, cudaMemcpyHostToDevice));
    }


    // --- 4. Halo Exchange for Conv1 ---
    // (Halo exchange code remains the same, including skip_halo1 label and bounds checks)
    if (haloRows1 > 0 && inputRowSizeBytes > 0) {
        MPI_Request send_req_up = MPI_REQUEST_NULL, recv_req_up = MPI_REQUEST_NULL;
        MPI_Request send_req_down = MPI_REQUEST_NULL, recv_req_down = MPI_REQUEST_NULL;
        std::vector<float> h_send_buffer_up(haloRows1 * W * C);
        std::vector<float> h_recv_buffer_up(haloRows1 * W * C);
        std::vector<float> h_send_buffer_down(haloRows1 * W * C);
        std::vector<float> h_recv_buffer_down(haloRows1 * W * C);

        int partner_up = rank - 1; int partner_down = rank + 1;
        int tag_up = 0; int tag_down = 1;
        int haloElementCount = inputHalo1SizeBytes / sizeof(float);
        if (haloElementCount <=0) goto skip_halo1;

        fprintf(stderr, "[Rank %d] Halo1: Exchanging %d elements (haloRows1=%d).\n", rank, haloElementCount, haloRows1);

        if (hasTopHalo) { MPI_Irecv(h_recv_buffer_up.data(), haloElementCount, MPI_FLOAT, partner_up, tag_up, MPI_COMM_WORLD, &recv_req_up); }
        if (hasBotHalo) { MPI_Irecv(h_recv_buffer_down.data(), haloElementCount, MPI_FLOAT, partner_down, tag_down, MPI_COMM_WORLD, &recv_req_down); }

        if (hasBotHalo && d_input_padded1 != nullptr) {
            size_t sendOffsetElements = inputOffsetElements + (localInputSizeBytes / sizeof(float)) - haloElementCount;
            if (sendOffsetElements * sizeof(float) >= 0 && sendOffsetElements + haloElementCount <= inputPadded1Size) { // Check non-negative offset and bounds
                 fprintf(stderr, "[Rank %d] Halo1 Send Up D->H: Offset=%" PRIu64 ", Count=%" PRIu64 "\n", rank, (uint64_t)sendOffsetElements, (uint64_t)haloElementCount);
                 CUDA_CHECK(cudaMemcpy(h_send_buffer_up.data(), d_input_padded1 + sendOffsetElements, inputHalo1SizeBytes, cudaMemcpyDeviceToHost));
                 MPI_Isend(h_send_buffer_up.data(), haloElementCount, MPI_FLOAT, partner_down, tag_up, MPI_COMM_WORLD, &send_req_up);
            } else { fprintf(stderr, "[Rank %d] Error: Halo1 send up offset invalid or out of bounds (Offset=%" PRId64 ", Count=%d, Size=%" PRIu64").\n", rank, (int64_t)sendOffsetElements, haloElementCount, (uint64_t)inputPadded1Size); MPI_Abort(MPI_COMM_WORLD, 1); }
        }
         if (hasTopHalo && d_input_padded1 != nullptr) {
             if (inputOffsetElements * sizeof(float) >= 0 && inputOffsetElements + haloElementCount <= inputPadded1Size) { // Check non-negative offset and bounds
                  fprintf(stderr, "[Rank %d] Halo1 Send Down D->H: Offset=%" PRIu64 ", Count=%" PRIu64 "\n", rank, (uint64_t)inputOffsetElements, (uint64_t)haloElementCount);
                  CUDA_CHECK(cudaMemcpy(h_send_buffer_down.data(), d_input_padded1 + inputOffsetElements, inputHalo1SizeBytes, cudaMemcpyDeviceToHost));
                  MPI_Isend(h_send_buffer_down.data(), haloElementCount, MPI_FLOAT, partner_up, tag_down, MPI_COMM_WORLD, &send_req_down);
             } else { fprintf(stderr, "[Rank %d] Error: Halo1 send down offset invalid or out of bounds (Offset=%" PRId64 ", Count=%d, Size=%" PRIu64").\n", rank, (int64_t)inputOffsetElements, haloElementCount, (uint64_t)inputPadded1Size); MPI_Abort(MPI_COMM_WORLD, 1); }
         }

        MPI_Status status;
        if (hasTopHalo) {
            MPI_Wait(&recv_req_up, &status);
             if (d_input_padded1 != nullptr && inputHalo1SizeBytes > 0) {
                 if (inputHalo1SizeBytes > inputPadded1Size * sizeof(float)) { fprintf(stderr, "[Rank %d] Error: Halo1 receive H->D dest buffer too small (top).\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
                 fprintf(stderr, "[Rank %d] Halo1 Recv Up H->D: DestOffset=0, Count=%" PRIu64 "\n", rank, (uint64_t)haloElementCount);
                 CUDA_CHECK(cudaMemcpy(d_input_padded1, h_recv_buffer_up.data(), inputHalo1SizeBytes, cudaMemcpyHostToDevice));
             }
             if (send_req_down != MPI_REQUEST_NULL) MPI_Wait(&send_req_down, &status);
        }
        if (hasBotHalo) {
            MPI_Wait(&recv_req_down, &status);
            size_t bottomHaloOffsetElements = inputOffsetElements + (localInputSizeBytes / sizeof(float));
             if (d_input_padded1 != nullptr && inputHalo1SizeBytes > 0 && bottomHaloOffsetElements + haloElementCount <= inputPadded1Size) {
                  fprintf(stderr, "[Rank %d] Halo1 Recv Down H->D: DestOffset=%" PRIu64 ", Count=%" PRIu64 "\n", rank, (uint64_t)bottomHaloOffsetElements, (uint64_t)haloElementCount);
                  CUDA_CHECK(cudaMemcpy(d_input_padded1 + bottomHaloOffsetElements, h_recv_buffer_down.data(), inputHalo1SizeBytes, cudaMemcpyHostToDevice));
             } else if (inputHalo1SizeBytes > 0) { fprintf(stderr, "[Rank %d] Error: Halo1 receive down offset out of bounds or null buffer (Offset=%" PRId64 ", Count=%d, Size=%" PRIu64").\n", rank, (int64_t)bottomHaloOffsetElements, haloElementCount, (uint64_t)inputPadded1Size); MPI_Abort(MPI_COMM_WORLD, 1); }
             if (send_req_up != MPI_REQUEST_NULL) MPI_Wait(&send_req_up, &status);
        }
    }
skip_halo1:;


    // --- 5. Execute Block 1 Kernels ---
     if (Hc1 > 0 && Wc1 > 0 && C1 > 0 && d_conv1_out && d_input_padded1 && d_weights1 && d_biases1) {
         fprintf(stderr, "[Rank %d] Launching Conv1: H=%d, W=%d, C=%d -> K=%d, F=%d, S=%d, P=%d\n", rank, paddedH1, W, C, p1.K, p1.F, p1.S, p1.P);
         cudaConvLayer(d_conv1_out, d_input_padded1, d_weights1, d_biases1, paddedH1, W, C, p1.K, p1.F, p1.S, p1.P);
         cudaReluLayer(d_conv1_out, conv1OutSize);
     }
     if (Hp1 > 0 && Wp1 > 0 && C1 > 0 && d_pool1_out && d_conv1_out) {
         fprintf(stderr, "[Rank %d] Launching Pool1: H=%d, W=%d, C=%d -> Fp=%d, Sp=%d\n", rank, Hc1, Wc1, C1, p1.F_pool, p1.S_pool);
         cudaMaxPoolLayer(d_pool1_out, d_conv1_out, Hc1, Wc1, C1, p1.F_pool, p1.S_pool);
     }


    // --- 6. Halo Exchange for Conv2 ---
    size_t pool1RowSizeBytes = (Wp1 > 0 && C1 > 0) ? (size_t)Wp1 * C1 * sizeof(float) : 0;
    if (haloRows2 > 0 && validHp1 > 0 && pool1RowSizeBytes > 0 && d_pool1_out) {
        MPI_Request send_req_up = MPI_REQUEST_NULL, recv_req_up = MPI_REQUEST_NULL;
        MPI_Request send_req_down = MPI_REQUEST_NULL, recv_req_down = MPI_REQUEST_NULL;

        size_t pool1HaloSizeBytes = (size_t)haloRows2 * pool1RowSizeBytes;
        if (haloRows2 > 0 && pool1RowSizeBytes > SIZE_MAX / (unsigned long)haloRows2) { fprintf(stderr, "[Rank %d] Error: pool1HaloSizeBytes calculation overflow.\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (pool1HaloSizeBytes == 0) goto skip_halo2;

        std::vector<float> h_send_buffer_up(pool1HaloSizeBytes / sizeof(float));
        std::vector<float> h_recv_buffer_up(pool1HaloSizeBytes / sizeof(float));
        std::vector<float> h_send_buffer_down(pool1HaloSizeBytes / sizeof(float));
        std::vector<float> h_recv_buffer_down(pool1HaloSizeBytes / sizeof(float));

        int partner_up = rank - 1; int partner_down = rank + 1;
        int tag_up = 2; int tag_down = 3;
        int haloElementCount = pool1HaloSizeBytes / sizeof(float);
        if (haloElementCount <= 0) goto skip_halo2;

        fprintf(stderr, "[Rank %d] Halo2: Exchanging %d elements (haloRows2=%d).\n", rank, haloElementCount, haloRows2);

        size_t pool1ValidDataOffsetBytes = (size_t)trimTop1 * pool1RowSizeBytes;
        if (pool1ValidDataOffsetBytes > pool1OutSize * sizeof(float)) { fprintf(stderr, "[Rank %d] Error: Halo2 trimTop1 offset (%" PRIu64 ") out of bounds (%" PRIu64 ").\n", rank, (uint64_t)pool1ValidDataOffsetBytes, (uint64_t)(pool1OutSize * sizeof(float))); MPI_Abort(MPI_COMM_WORLD, 1); }
        float* d_pool1_valid_start = d_pool1_out + pool1ValidDataOffsetBytes / sizeof(float);
        size_t pool1ValidDataSizeBytes = (size_t)validHp1 * pool1RowSizeBytes;
         if (pool1ValidDataOffsetBytes + pool1ValidDataSizeBytes > pool1OutSize * sizeof(float)) { fprintf(stderr, "[Rank %d] Error: Halo2 valid data region (offset %" PRIu64 ", size %" PRIu64 ") exceeds buffer bounds (%" PRIu64 ").\n", rank, (uint64_t)pool1ValidDataOffsetBytes, (uint64_t)pool1ValidDataSizeBytes, (uint64_t)(pool1OutSize * sizeof(float))); MPI_Abort(MPI_COMM_WORLD, 1); }

        if (hasTopHalo2) { MPI_Irecv(h_recv_buffer_up.data(), haloElementCount, MPI_FLOAT, partner_up, tag_up, MPI_COMM_WORLD, &recv_req_up); }
        if (hasBotHalo2) { MPI_Irecv(h_recv_buffer_down.data(), haloElementCount, MPI_FLOAT, partner_down, tag_down, MPI_COMM_WORLD, &recv_req_down); }

        if (hasBotHalo2) { // Send my bottom rows up
            size_t sendOffsetBytes = pool1ValidDataSizeBytes - pool1HaloSizeBytes;
             if (pool1HaloSizeBytes > pool1ValidDataSizeBytes) { // || sendOffsetBytes > pool1ValidDataSizeBytes // Offset check implicit in next check
                 fprintf(stderr, "[Rank %d] Error: Halo2 send offset calculation error (bottom). ValidSize=%" PRIu64 ", HaloSize=%" PRIu64 "\n", rank, (uint64_t)pool1ValidDataSizeBytes, (uint64_t)pool1HaloSizeBytes); MPI_Abort(MPI_COMM_WORLD, 1);
             }
             if (pool1ValidDataOffsetBytes + sendOffsetBytes + pool1HaloSizeBytes > pool1OutSize * sizeof(float)) { fprintf(stderr, "[Rank %d] Error: Halo2 send D->H source out of bounds (bottom).\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
            fprintf(stderr, "[Rank %d] Halo2 Send Up D->H: Offset=%" PRIu64 ", Count=%" PRIu64 "\n", rank, (uint64_t)(pool1ValidDataOffsetBytes/sizeof(float) + sendOffsetBytes/sizeof(float)), (uint64_t)haloElementCount);
            CUDA_CHECK(cudaMemcpy(h_send_buffer_up.data(), d_pool1_valid_start + sendOffsetBytes / sizeof(float), pool1HaloSizeBytes, cudaMemcpyDeviceToHost));
            MPI_Isend(h_send_buffer_up.data(), haloElementCount, MPI_FLOAT, partner_down, tag_up, MPI_COMM_WORLD, &send_req_up);
        }
         if (hasTopHalo2) { // Send my top rows down
              if (pool1HaloSizeBytes > pool1ValidDataSizeBytes) { fprintf(stderr, "[Rank %d] Error: Halo2 send D->H source size error (top - halo > valid). ValidSize=%" PRIu64 ", HaloSize=%" PRIu64 "\n", rank, (uint64_t)pool1ValidDataSizeBytes, (uint64_t)pool1HaloSizeBytes); MPI_Abort(MPI_COMM_WORLD, 1); }
               if (pool1ValidDataOffsetBytes + pool1HaloSizeBytes > pool1OutSize * sizeof(float)) { fprintf(stderr, "[Rank %d] Error: Halo2 send D->H source out of bounds (top - offset+halo > total).\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
             fprintf(stderr, "[Rank %d] Halo2 Send Down D->H: Offset=%" PRIu64 ", Count=%" PRIu64 "\n", rank, (uint64_t)(pool1ValidDataOffsetBytes/sizeof(float)), (uint64_t)haloElementCount);
             CUDA_CHECK(cudaMemcpy(h_send_buffer_down.data(), d_pool1_valid_start, pool1HaloSizeBytes, cudaMemcpyDeviceToHost));
             MPI_Isend(h_send_buffer_down.data(), haloElementCount, MPI_FLOAT, partner_up, tag_down, MPI_COMM_WORLD, &send_req_down);
         }

        size_t input2CurrentOffsetBytes = 0;
        if (hasTopHalo2) {
             MPI_Wait(&recv_req_up, MPI_STATUS_IGNORE);
             if (d_input_padded2 != nullptr && pool1HaloSizeBytes > 0 && pool1HaloSizeBytes <= inputPadded2Size * sizeof(float)) {
                  fprintf(stderr, "[Rank %d] Halo2 Recv Up H->D: DestOffset=0, Count=%" PRIu64 "\n", rank, (uint64_t)haloElementCount);
                  CUDA_CHECK(cudaMemcpy(d_input_padded2, h_recv_buffer_up.data(), pool1HaloSizeBytes, cudaMemcpyHostToDevice));
             } else if (pool1HaloSizeBytes > 0) { fprintf(stderr, "[Rank %d] Error: Halo2 receive H->D dest out of bounds or null buffer (top).\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
             input2CurrentOffsetBytes += pool1HaloSizeBytes;
             if (send_req_down != MPI_REQUEST_NULL) MPI_Wait(&send_req_down, MPI_STATUS_IGNORE);
        }
        if (validHp1 > 0 && pool1ValidDataSizeBytes > 0 && d_input_padded2 != nullptr) {
            if (input2CurrentOffsetBytes + pool1ValidDataSizeBytes > inputPadded2Size * sizeof(float)) { fprintf(stderr, "[Rank %d] Error: Halo2 D->D copy destination out of bounds. Offset=%" PRIu64 ", CopySize=%" PRIu64 ", BufferSize=%" PRIu64 "\n", rank, (uint64_t)input2CurrentOffsetBytes, (uint64_t)pool1ValidDataSizeBytes, (uint64_t)(inputPadded2Size * sizeof(float))); MPI_Abort(MPI_COMM_WORLD, 1); }
             fprintf(stderr, "[Rank %d] Halo2 Copy Valid D->D: DestOffset=%" PRIu64 ", SrcOffset=%" PRIu64 ", Count=%" PRIu64 "\n", rank, (uint64_t)(input2CurrentOffsetBytes / sizeof(float)), (uint64_t)(pool1ValidDataOffsetBytes / sizeof(float)), (uint64_t)(pool1ValidDataSizeBytes / sizeof(float)));
             CUDA_CHECK(cudaMemcpy(d_input_padded2 + input2CurrentOffsetBytes / sizeof(float), d_pool1_valid_start, pool1ValidDataSizeBytes, cudaMemcpyDeviceToDevice));
             input2CurrentOffsetBytes += pool1ValidDataSizeBytes;
        }
        if (hasBotHalo2) {
            MPI_Wait(&recv_req_down, MPI_STATUS_IGNORE);
             if (d_input_padded2 != nullptr && pool1HaloSizeBytes > 0 && input2CurrentOffsetBytes + pool1HaloSizeBytes <= inputPadded2Size * sizeof(float)) {
                  fprintf(stderr, "[Rank %d] Halo2 Recv Down H->D: DestOffset=%" PRIu64 ", Count=%" PRIu64 "\n", rank, (uint64_t)(input2CurrentOffsetBytes / sizeof(float)), (uint64_t)haloElementCount);
                  CUDA_CHECK(cudaMemcpy(d_input_padded2 + input2CurrentOffsetBytes / sizeof(float), h_recv_buffer_down.data(), pool1HaloSizeBytes, cudaMemcpyHostToDevice));
             } else if (pool1HaloSizeBytes > 0) { fprintf(stderr, "[Rank %d] Error: Halo2 receive H->D dest out of bounds or null buffer (bottom).\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
             if (send_req_up != MPI_REQUEST_NULL) MPI_Wait(&send_req_up, MPI_STATUS_IGNORE);
        }

    } else if (validHp1 > 0 && pool1RowSizeBytes > 0 && d_pool1_out && d_input_padded2) {
        // Case: No halo exchange needed, copy valid data directly D->D
        size_t pool1ValidDataOffsetBytes = (size_t)trimTop1 * pool1RowSizeBytes;
        float* d_pool1_valid_start = d_pool1_out + pool1ValidDataOffsetBytes / sizeof(float);
        size_t pool1ValidDataSizeBytes = (size_t)validHp1 * pool1RowSizeBytes;
         if (pool1ValidDataOffsetBytes + pool1ValidDataSizeBytes > pool1OutSize * sizeof(float)) { fprintf(stderr, "[Rank %d] Error: Halo2 D->D source bounds error (no halo case).\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
         if (pool1ValidDataSizeBytes > inputPadded2Size * sizeof(float)) { fprintf(stderr, "[Rank %d] Error: Halo2 D->D destination bounds error (no halo case).\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (pool1ValidDataSizeBytes > 0) {
            fprintf(stderr, "[Rank %d] Halo2 Copy Valid D->D (No Halo): DestOffset=0, SrcOffset=%" PRIu64 ", Count=%" PRIu64 "\n", rank, (uint64_t)(pool1ValidDataOffsetBytes / sizeof(float)), (uint64_t)(pool1ValidDataSizeBytes / sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_input_padded2, d_pool1_valid_start, pool1ValidDataSizeBytes, cudaMemcpyDeviceToDevice));
        }
    } else if (d_input_padded2 && inputPadded2Size > 0) {
        // Case: No valid input data, zero out the buffer for layer 2
        fprintf(stderr, "[Rank %d] Halo2: Zeroing d_input_padded2 (size %" PRIu64 ").\n", rank, (uint64_t)inputPadded2Size);
        CUDA_CHECK(cudaMemset(d_input_padded2, 0, inputPadded2Size * sizeof(float)));
    }
skip_halo2:;


    // --- 7. Execute Block 2 Kernels ---
     if (Hc2 > 0 && Wc2 > 0 && C2 > 0 && d_conv2_out && d_input_padded2 && d_weights2 && d_biases2) {
        fprintf(stderr, "[Rank %d] Launching Conv2: H=%d, W=%d, C=%d -> K=%d, F=%d, S=%d, P=%d\n", rank, paddedH2, Wp1, C1, p2.K, p2.F, p2.S, p2.P);
        cudaConvLayer(d_conv2_out, d_input_padded2, d_weights2, d_biases2, paddedH2, Wp1, C1, p2.K, p2.F, p2.S, p2.P);
        cudaReluLayer(d_conv2_out, conv2OutSize);
    }
     if (Hp2 > 0 && Wp2 > 0 && C2 > 0 && d_pool2_out && d_conv2_out) {
        fprintf(stderr, "[Rank %d] Launching Pool2: H=%d, W=%d, C=%d -> Fp=%d, Sp=%d\n", rank, Hc2, Wc2, C2, p2.F_pool, p2.S_pool);
        cudaMaxPoolLayer(d_pool2_out, d_conv2_out, Hc2, Wc2, C2, p2.F_pool, p2.S_pool);
     }
     if (Hp2 > 0 && Wp2 > 0 && C2 > 0 && d_lrn2_out && d_pool2_out) {
        fprintf(stderr, "[Rank %d] Launching LRN2: H=%d, W=%d, C=%d\n", rank, Hp2, Wp2, C2);
        cudaLRNLayer(d_lrn2_out, d_pool2_out, Hp2, Wp2, C2, p2.N_lrn, p2.alpha, p2.beta, p2.k_lrn);
     }


    // --- 8. Copy Final Result Device -> Host ---
    size_t finalRowSizeBytes = (Wp2 > 0 && C2 > 0) ? (size_t)Wp2 * C2 * sizeof(float) : 0;
    size_t finalLocalSizeBytes = (size_t)finalLocalH * finalRowSizeBytes;

    if (finalLocalSizeBytes > 0 && finalLocalH > 0) {
         // Check for potential overflow before resize
        if ((unsigned long)finalLocalH > SIZE_MAX / ((unsigned long)Wp2*C2) ) {
             fprintf(stderr, "[Rank %d] Error: Final output size calculation overflow before resize.\n", rank); MPI_Abort(MPI_COMM_WORLD, 1);
        }
         h_localOutput.resize(finalLocalSizeBytes / sizeof(float));
    } else {
         h_localOutput.clear();
         finalLocalSizeBytes = 0;
    }

    if (finalLocalH > 0 && finalLocalSizeBytes > 0 && d_lrn2_out != nullptr) {
        size_t finalDataOffsetBytes = (size_t)trimTop2 * finalRowSizeBytes;
        if (finalDataOffsetBytes + finalLocalSizeBytes > lrn2OutSize * sizeof(float)) {
             fprintf(stderr, "[Rank %d] Error: Final D->H copy source (offset %" PRIu64 ", size %" PRIu64 ") out of bounds (%" PRIu64 "). finalLocalH=%d, trimTop2=%d, Hp2=%d, Wp2=%d, C2=%d\n", rank, (uint64_t)finalDataOffsetBytes, (uint64_t)finalLocalSizeBytes, (uint64_t)(lrn2OutSize * sizeof(float)), finalLocalH, trimTop2, Hp2, Wp2, C2); MPI_Abort(MPI_COMM_WORLD, 1);
          }
        fprintf(stderr, "[Rank %d] Final D->H: DestSize=%" PRIu64 ", SrcOffset=%" PRIu64 ", CopySize=%" PRIu64 "\n", rank, (uint64_t)h_localOutput.size(), (uint64_t)(finalDataOffsetBytes / sizeof(float)), (uint64_t)(finalLocalSizeBytes / sizeof(float)));
        CUDA_CHECK(cudaMemcpy(h_localOutput.data(), d_lrn2_out + finalDataOffsetBytes / sizeof(float), finalLocalSizeBytes, cudaMemcpyDeviceToHost));
    }

    // --- 9. Free Device Memory ---
    if (d_input_padded1) cudaFree(d_input_padded1);
    if (d_conv1_out)     cudaFree(d_conv1_out);
    if (d_pool1_out)     cudaFree(d_pool1_out);
    if (d_input_padded2) cudaFree(d_input_padded2);
    if (d_conv2_out)     cudaFree(d_conv2_out);
    if (d_pool2_out)     cudaFree(d_pool2_out);
    if (d_lrn2_out)      cudaFree(d_lrn2_out);
    if (d_weights1)      cudaFree(d_weights1);
    if (d_biases1)       cudaFree(d_biases1);
    if (d_weights2)      cudaFree(d_weights2);
    if (d_biases2)       cudaFree(d_biases2);

} // End of alexnetForwardPassMPI_CUDA definition
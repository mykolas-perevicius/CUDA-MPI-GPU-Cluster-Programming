// final_project/v2_mpi_only/2.2_scatter_halo/src/main.cpp
#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip> // For std::fixed, std::setprecision if needed
#include <cmath>   // Needed for std::max
#include <chrono>  // For timing

#include "../include/alexnet.hpp" // Includes LayerParams and inline helpers
#include "../include/layers.hpp"  // Includes serial layer function prototypes

// 3D index helper (local definition ok)
inline size_t idx3D(int h, int w, int c, int W, int C) {
    // Add checks for h, w, c boundaries if needed for robustness
    return (static_cast<size_t>(h) * W + w) * C + c;
}

// Note: convOutDim and poolOutDim are now defined inline in alexnet.hpp


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1) Common setup
    int H = 227, W = 227, C = 3;
    LayerParams conv1, conv2;
    std::vector<float> input, finalOut;
    int finalH = -1, finalW = -1, finalC = -1;

    if (rank == 0) {
        // initialize input & params
        input.assign((size_t)H * W * C, 1.0f);
        conv1.K = 96; conv1.F = 11; conv1.S = 4; conv1.P = 0;
        conv1.F_pool = 3; conv1.S_pool = 2;
        conv1.weights.assign((size_t)conv1.K * C * conv1.F * conv1.F, 0.01f);
        conv1.biases.assign(conv1.K, 0.0f);
        conv2.K = 256; conv2.F = 5; conv2.S = 1; conv2.P = 2;
        conv2.F_pool = 3; conv2.S_pool = 2;
        conv2.N_lrn = 5; conv2.alpha = 1e-4f; conv2.beta = 0.75f; conv2.k_lrn = 2.0f;
        int C_conv2_input = conv1.K;
        conv2.weights.assign((size_t)conv2.K * C_conv2_input * conv2.F * conv2.F, 0.01f);
        conv2.biases.assign(conv2.K, 0.0f);

        // Calculate expected final dimensions (using inline helpers from alexnet.hpp)
        int H1 = convOutDim(H, conv1.F, conv1.P, conv1.S);
        int W1 = convOutDim(W, conv1.F, conv1.P, conv1.S);
        H1 = poolOutDim(H1, conv1.F_pool, conv1.S_pool);
        W1 = poolOutDim(W1, conv1.F_pool, conv1.S_pool);
        int H2 = convOutDim(H1, conv2.F, conv2.P, conv2.S);
        int W2 = convOutDim(W1, conv2.F, conv2.P, conv2.S);
        H2 = poolOutDim(H2, conv2.F_pool, conv2.S_pool);
        W2 = poolOutDim(W2, conv2.F_pool, conv2.S_pool);
        finalH = H2; finalW = W2; finalC = conv2.K;
    }

    // 2) Broadcast sizes & params
    auto bcastInt = [&](int& x){ MPI_Bcast(&x,1,MPI_INT,0,MPI_COMM_WORLD); };
    bcastInt(H); bcastInt(W); bcastInt(C);
    bcastInt(conv1.K); bcastInt(conv1.F); bcastInt(conv1.S); bcastInt(conv1.P);
    bcastInt(conv1.F_pool); bcastInt(conv1.S_pool);
    bcastInt(conv2.K); bcastInt(conv2.F); bcastInt(conv2.S); bcastInt(conv2.P);
    bcastInt(conv2.F_pool); bcastInt(conv2.S_pool);
    bcastInt(conv2.N_lrn);
    bcastInt(finalH); bcastInt(finalW); bcastInt(finalC);

    if (rank != 0) {
        input.resize((size_t)H * W * C);
        conv1.weights.resize((size_t)conv1.K * C * conv1.F * conv1.F);
        conv1.biases.resize(conv1.K);
        int C_conv2_input = conv1.K;
        conv2.weights.resize((size_t)conv2.K * C_conv2_input * conv2.F * conv2.F);
        conv2.biases.resize(conv2.K);
    }
    float lrnArr[3];
    if (rank == 0) { lrnArr[0] = conv2.alpha; lrnArr[1] = conv2.beta; lrnArr[2] = conv2.k_lrn; }
    MPI_Bcast(lrnArr,3,MPI_FLOAT,0,MPI_COMM_WORLD);
    if (rank != 0) { conv2.alpha = lrnArr[0]; conv2.beta = lrnArr[1]; conv2.k_lrn = lrnArr[2]; }
    MPI_Bcast(input.data(),           static_cast<int>(input.size()),           MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(conv1.weights.data(),   static_cast<int>(conv1.weights.size()),   MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(conv1.biases.data(),    static_cast<int>(conv1.biases.size()),    MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(conv2.weights.data(),   static_cast<int>(conv2.weights.size()),   MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(conv2.biases.data(),    static_cast<int>(conv2.biases.size()),    MPI_FLOAT,0,MPI_COMM_WORLD);

    // --- Barrier before starting main computation (optional) ---
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_start_compute = std::chrono::high_resolution_clock::now(); // *** Start Compute Timer ***

    // Path A: broadcast-all if only one rank (np=1 case)
    if (size == 1) {
        // Definition is in alexnet_mpi.cpp
        alexnetForwardPassMPI(input, conv1, conv2, H, W, C, finalOut);
    }
    // Path B: scatter + halo (np > 1 case)
    else {
        // 3) Scatter input
        std::vector<int> sendCnt(size), sendDisp(size);
        int base = H / size, rem = H % size;
        if (rank == 0) {
            sendDisp[0] = 0;
            for (int i = 0; i < size; ++i) {
                int rows = base + (i < rem ? 1 : 0);
                sendCnt[i]  = rows * W * C;
                if (i > 0) { sendDisp[i] = sendDisp[i - 1] + sendCnt[i - 1]; }
            }
        }
        int localCnt;
        MPI_Scatter(sendCnt.data(), 1, MPI_INT, &localCnt, 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<float> localIn(localCnt);
        MPI_Scatterv(input.data(), sendCnt.data(), sendDisp.data(), MPI_FLOAT,
                     localIn.data(), localCnt, MPI_FLOAT, 0, MPI_COMM_WORLD);
        int localH = (W*C > 0) ? localCnt / (W * C) : 0;

        // 4) Halo for Conv1
        const int pad1 = conv1.F / 2;
        int slice1 = 0; if (pad1 > 0 && W > 0 && C > 0) slice1 = pad1 * W * C;
        std::vector<float> topHalo(slice1), botHalo(slice1);
        MPI_Request send_req_up = MPI_REQUEST_NULL, recv_req_up = MPI_REQUEST_NULL;
        MPI_Request send_req_down = MPI_REQUEST_NULL, recv_req_down = MPI_REQUEST_NULL;
        if (pad1 > 0) {
            if (rank > 0) {
                MPI_Irecv(topHalo.data(), slice1, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, &recv_req_up);
                MPI_Isend(localIn.data(), slice1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &send_req_up);
            } else { std::fill(topHalo.begin(), topHalo.end(), 0.0f); }
            if (rank < size - 1) {
                MPI_Irecv(botHalo.data(), slice1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &recv_req_down);
                MPI_Isend(localIn.data() + (size_t)std::max(0, localH - pad1) * W * C, slice1, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, &send_req_down);
            } else { std::fill(botHalo.begin(), botHalo.end(), 0.0f); }
            MPI_Wait(&recv_req_up, MPI_STATUS_IGNORE); MPI_Wait(&send_req_up, MPI_STATUS_IGNORE);
            MPI_Wait(&recv_req_down, MPI_STATUS_IGNORE); MPI_Wait(&send_req_down, MPI_STATUS_IGNORE);
        }

        // 5) Conv1 → ReLU1 → Pool1
        int pH1 = localH + 2 * pad1;
        std::vector<float> padded1((size_t)std::max(0,pH1) * W * C);
        if (pad1 > 0 && localIn.size() > 0) { // Check localIn isn't empty
             if (slice1 <= static_cast<int>(padded1.size())) std::copy(topHalo.begin(), topHalo.end(), padded1.begin());
             if (slice1 + localIn.size() <= padded1.size()) std::copy(localIn.begin(), localIn.end(), padded1.begin() + slice1);
             if (slice1 + localIn.size() + slice1 <= padded1.size()) std::copy(botHalo.begin(), botHalo.end(), padded1.begin() + slice1 + localIn.size());
        } else { padded1 = localIn; }

        int Hc1 = convOutDim(pH1, conv1.F, conv1.P, conv1.S);
        int Wc1 = convOutDim(W,   conv1.F, conv1.P, conv1.S);
        std::vector<float> c1out((size_t)std::max(0,Hc1) * Wc1 * conv1.K);
        // Ensure layers are defined (layers_mpi.cpp) - V2 uses different layer signatures!
        serialConvLayer(c1out, padded1, conv1, pH1, W, C); // V2 uses LayerParams struct
        serialReluLayer(c1out);

        int Hp1 = poolOutDim(Hc1, conv1.F_pool, conv1.S_pool);
        int Wp1 = poolOutDim(Wc1, conv1.F_pool, conv1.S_pool);
        std::vector<float> pool1_p((size_t)std::max(0,Hp1) * Wp1 * conv1.K);
        serialMaxPoolLayer(pool1_p, c1out, Hc1, Wc1, conv1.K, conv1.F_pool, conv1.S_pool);


        // 6) Asymmetric trim of pool1_p (Corrected Logic)
        int trim_top1 = 0, trim_bot1 = 0;
        if (pad1 > 0 && conv1.S > 0 && conv1.S_pool > 0) {
            int rows_affected_by_pad_after_conv1 = (pad1 + conv1.S - 1) / conv1.S;
            int trim_amount1 = (rows_affected_by_pad_after_conv1 + conv1.S_pool - 1) / conv1.S_pool;
            trim_top1 = (rank > 0        ) ? trim_amount1 : 0;
            trim_bot1 = (rank < size - 1 ) ? trim_amount1 : 0;
        }
        if (trim_top1 + trim_bot1 >= Hp1 && Hp1 > 0) { if(rank==0) std::cerr<<"E1 "; MPI_Abort(MPI_COMM_WORLD, 1); }
        int realHp1 = Hp1 - trim_top1 - trim_bot1;
        std::vector<float> pool1Out((size_t)std::max(0, realHp1) * Wp1 * conv1.K);
        if (realHp1 > 0) {
            for (int r = 0; r < realHp1; ++r) {
                 size_t src_off = (size_t)(r + trim_top1) * Wp1 * conv1.K; size_t dst_off = (size_t)r * Wp1 * conv1.K; size_t count = (size_t)Wp1 * conv1.K;
                 if (src_off + count <= pool1_p.size() && dst_off + count <= pool1Out.size()) { std::copy_n(pool1_p.data() + src_off, count, pool1Out.data() + dst_off); }
                 else { if(rank==0) std::cerr<<"E2 "; MPI_Abort(MPI_COMM_WORLD, 1); }
            }
        } else { pool1Out.clear(); realHp1 = 0; }

        // 7) Halo Exchange for Conv2
        int C1 = conv1.K; int pad2 = conv2.F / 2; int slice2 = 0; if(pad2 > 0 && Wp1 > 0 && C1 > 0) slice2 = pad2 * Wp1 * C1;
        std::vector<float> topHalo2(slice2), botHalo2(slice2);
        MPI_Request send_req2_up = MPI_REQUEST_NULL, recv_req2_up = MPI_REQUEST_NULL;
        MPI_Request send_req2_down = MPI_REQUEST_NULL, recv_req2_down = MPI_REQUEST_NULL;
        if (pad2 > 0 && realHp1 > 0) {
            if (rank > 0) { MPI_Irecv(topHalo2.data(), slice2, MPI_FLOAT, rank - 1, 3, MPI_COMM_WORLD, &recv_req2_up); int send_count_up = std::min(slice2, (int)pool1Out.size()); if(send_count_up > 0) MPI_Isend(pool1Out.data(), send_count_up, MPI_FLOAT, rank - 1, 2, MPI_COMM_WORLD, &send_req2_up); else send_req2_up = MPI_REQUEST_NULL;} else { std::fill(topHalo2.begin(), topHalo2.end(), 0.0f); }
            if (rank < size - 1) { MPI_Irecv(botHalo2.data(), slice2, MPI_FLOAT, rank + 1, 2, MPI_COMM_WORLD, &recv_req2_down); size_t send_offset_down = (size_t)std::max(0, realHp1 - pad2) * Wp1 * C1; int send_count_down = std::min(slice2, (int)((size_t)realHp1*Wp1*C1 - send_offset_down)); if (send_count_down > 0) { MPI_Isend(pool1Out.data() + send_offset_down, send_count_down, MPI_FLOAT, rank + 1, 3, MPI_COMM_WORLD, &send_req2_down); } else { send_req2_down = MPI_REQUEST_NULL; } } else { std::fill(botHalo2.begin(), botHalo2.end(), 0.0f); }
            MPI_Wait(&recv_req2_up, MPI_STATUS_IGNORE); MPI_Wait(&send_req2_up, MPI_STATUS_IGNORE); MPI_Wait(&recv_req2_down, MPI_STATUS_IGNORE); MPI_Wait(&send_req2_down, MPI_STATUS_IGNORE);
        } else { std::fill(topHalo2.begin(), topHalo2.end(), 0.0f); std::fill(botHalo2.begin(), botHalo2.end(), 0.0f); }

        int pH2 = realHp1 + 2 * pad2;
        std::vector<float> padded2((size_t)std::max(0,pH2) * Wp1 * C1, 0.0f);
        if (realHp1 > 0 && pool1Out.size() > 0) { // Check pool1Out not empty
             if (pad2 > 0) {
                 if (slice2 <= static_cast<int>(padded2.size())) std::copy(topHalo2.begin(), topHalo2.end(), padded2.begin());
                 if (slice2 + pool1Out.size() <= padded2.size()) std::copy(pool1Out.begin(), pool1Out.end(), padded2.begin() + slice2);
                 if (slice2 + pool1Out.size() + slice2 <= padded2.size()) std::copy(botHalo2.begin(), botHalo2.end(), padded2.begin() + slice2 + pool1Out.size());
             } else { padded2 = pool1Out; }
        }

        int Hc2 = convOutDim(pH2, conv2.F, conv2.P, conv2.S);
        int Wc2 = convOutDim(Wp1, conv2.F, conv2.P, conv2.S);
        std::vector<float> c2out((size_t)std::max(0,Hc2) * Wc2 * conv2.K);
        serialConvLayer(c2out, padded2, conv2, pH2, Wp1, C1); // V2 uses LayerParams struct
        serialReluLayer(c2out);

        int Hp2 = poolOutDim(Hc2, conv2.F_pool, conv2.S_pool);
        int Wp2 = poolOutDim(Wc2, conv2.F_pool, conv2.S_pool);
        std::vector<float> p2out((size_t)std::max(0,Hp2) * Wp2 * conv2.K);
        serialMaxPoolLayer(p2out, c2out, Hc2, Wc2, conv2.K, conv2.F_pool, conv2.S_pool);

        std::vector<float> l2out(p2out.size());
        serialLRNLayer(l2out, p2out, Hp2, Wp2, conv2.K, conv2.N_lrn, conv2.alpha, conv2.beta, conv2.k_lrn);

        // 8) Trim output of LRN2 (Corrected Logic)
        int trim_top2 = 0, trim_bot2 = 0;
        if (pad2 > 0 && conv2.S > 0 && conv2.S_pool > 0) {
            int rows_affected_by_pad_after_conv2 = (pad2 + conv2.S - 1) / conv2.S;
            int trim_amount2 = (rows_affected_by_pad_after_conv2 + conv2.S_pool - 1) / conv2.S_pool;
            trim_top2 = (rank > 0        ) ? trim_amount2 : 0;
            trim_bot2 = (rank < size - 1 ) ? trim_amount2 : 0;
        }
        if (trim_top2 + trim_bot2 >= Hp2 && Hp2 > 0) { if(rank==0) std::cerr<<"E3 "; MPI_Abort(MPI_COMM_WORLD, 1); }
        int realHp2 = Hp2 - trim_top2 - trim_bot2;
        std::vector<float> localOut((size_t)std::max(0, realHp2) * Wp2 * conv2.K);
         if (realHp2 > 0) {
            for (int r = 0; r < realHp2; ++r) {
                 size_t src_off = (size_t)(r + trim_top2) * Wp2 * conv2.K; size_t dst_off = (size_t)r * Wp2 * conv2.K; size_t count = (size_t)Wp2 * conv2.K;
                 if (src_off + count <= l2out.size() && dst_off + count <= localOut.size()) { std::copy_n(l2out.data() + src_off, count, localOut.data() + dst_off); }
                 else { if(rank==0) std::cerr<<"E4 "; MPI_Abort(MPI_COMM_WORLD, 1); }
            }
         } else { localOut.clear(); realHp2 = 0; }

        // Gather sizes
        int outCnt = static_cast<int>(localOut.size());
        std::vector<int> recvCnt(size);
        MPI_Gather(&outCnt, 1, MPI_INT, recvCnt.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate displacements and total size on Rank 0
        std::vector<int> recvDisp;
        int totalFinalSize = 0; // Define here for rank 0 scope
        if (rank == 0) {
            recvDisp.resize(size); recvDisp[0] = 0; totalFinalSize = recvCnt[0];
            for (int i = 1; i < size; ++i) { recvDisp[i] = recvDisp[i-1] + recvCnt[i-1]; totalFinalSize += recvCnt[i]; }
            finalOut.resize(totalFinalSize);
        }

        // Gather final data
        MPI_Gatherv(localOut.data(), outCnt, MPI_FLOAT,
                    finalOut.data(), recvCnt.data(), recvDisp.data(),
                    MPI_FLOAT, 0, MPI_COMM_WORLD);
        // --- End of Scatter/Halo Path Computation ---
    } // End else (size > 1)

    // --- Barrier after all computation and gathering ---
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = std::chrono::high_resolution_clock::now(); // End Timer here

    // 9) Final detailed summary and time (on rank 0)
    if (rank == 0) {
        // Calculate duration
        double duration_ms = std::chrono::duration<double, std::milli>(t_end - t_start_compute).count();

        // Verify final dimensions if size > 1 (check against broadcasted finalH/W/C)
        int expectedTotalSize = finalH * finalW * finalC;
        // Need totalFinalSize if size > 1; it's only calculated in the else block.
        // Recalculate or retrieve totalFinalSize if needed here. Simpler: check finalOut size.
        if (finalOut.size() != static_cast<size_t>(expectedTotalSize) && size > 1) {
              // Note: This check might fail if finalH/W/C were incorrectly calculated initially.
              // It's mainly a sanity check for the gather process.
             std::cerr << "WARNING: Gathered size (" << finalOut.size()
                       << ") does not match expected final size ("
                       << finalH << "x" << finalW << "x" << finalC << " = " << expectedTotalSize
                       << "). Check trim/halo/layer logic." << std::endl;
        }

        // Print shape and sample values
        std::cout << "shape: " << finalH << "x" << finalW << "x" << finalC << std::endl;
        std::cout << "Sample values: ";
        int num_to_print = std::min((size_t)5, finalOut.size());
        for (int i = 0; i < num_to_print; ++i) {
            std::cout << finalOut[i] << (i == num_to_print - 1 ? "" : " ");
        }
        std::cout << std::endl;

         // Print time in expected format
        std::cout << "Execution Time: " << duration_ms << " ms" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
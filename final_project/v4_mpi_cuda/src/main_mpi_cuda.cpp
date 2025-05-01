// === final_project/v4_mpi_cuda/src/main_mpi_cuda.cpp ===
#include "alexnet.hpp"
#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>

#define CUDA_CHECK(x) do{cudaError_t e=(x);                     \
  if(e!=cudaSuccess){int r;MPI_Comm_rank(MPI_COMM_WORLD,&r);    \
    std::cerr<<"[Rank "<<r<<"] "<<cudaGetErrorString(e)<<"\n";  \
    MPI_Abort(MPI_COMM_WORLD,1);} }while(0)

/*----------------------------------------------------------------*/
static void runHybridMPI(int H,int W,int C,
                         LayerParams& p1,LayerParams& p2)
{
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    /* ------------ root initialises full input ------------------ */
    std::vector<float> full;
    if(rank==0){
        full.assign((size_t)H*W*C,1.f);
        p1.weights.assign((size_t)p1.K*C*p1.F*p1.F,0.01f);
        p1.biases .assign(p1.K,0.f);
        p2.weights.assign((size_t)p2.K*p1.K*p2.F*p2.F,0.01f);
        p2.biases .assign(p2.K,0.f);
    }

    /* ------------ broadcast hyper-parameters ------------------- */
    auto bI=[&](int& v){ MPI_Bcast(&v,1,MPI_INT ,0,MPI_COMM_WORLD); };
    auto bF=[&](float& v){ MPI_Bcast(&v,1,MPI_FLOAT,0,MPI_COMM_WORLD); };

    bI(p1.K);bI(p1.F);bI(p1.S);bI(p1.P);bI(p1.F_pool);bI(p1.S_pool);
    bI(p2.K);bI(p2.F);bI(p2.S);bI(p2.P);bI(p2.F_pool);bI(p2.S_pool);
    bI(p2.N_lrn);bF(p2.alpha);bF(p2.beta);bF(p2.k_lrn);

    if(rank!=0){
        p1.weights.resize((size_t)p1.K*C*p1.F*p1.F);
        p1.biases .resize(p1.K);
        p2.weights.resize((size_t)p2.K*p1.K*p2.F*p2.F);
        p2.biases .resize(p2.K);
    }
    MPI_Bcast(p1.weights.data(),(int)p1.weights.size(),MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(p1.biases .data(),(int)p1.biases .size(),MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(p2.weights.data(),(int)p2.weights.size(),MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(p2.biases .data(),(int)p2.biases .size(),MPI_FLOAT,0,MPI_COMM_WORLD);

    /* ------------ scatter rows --------------------------------- */
    std::vector<int> rows(size),cnt(size),disp(size);
    if(rank==0){
        int base=H/size,rem=H%size,off=0;
        for(int r=0;r<size;++r){
            rows[r]=base+(r<rem);
            cnt [r]=rows[r]*W*C; disp[r]=off; off+=cnt[r];
        }
    }
    int myRows;
    MPI_Scatter(rows.data(),1,MPI_INT,&myRows,1,MPI_INT,0,MPI_COMM_WORLD);

    std::vector<float> myIn(myRows*W*C);
    MPI_Scatterv(full.data(),cnt.data(),disp.data(),MPI_FLOAT,
                 myIn.data(),(int)myIn.size(),MPI_FLOAT,0,MPI_COMM_WORLD);

    /* ------------ choose GPU ----------------------------------- */
    int nDev=0; cudaGetDeviceCount(&nDev);
    CUDA_CHECK(cudaSetDevice(rank % std::max(1,nDev)));

    /* ------------ top/bottom halo exchange (Conv1) ------------- */
    int halo=p1.F/2;                      // rows
    int rowSz=W*C;                        // elements
    if(halo){
        std::vector<float> recvT(halo*rowSz),recvB(halo*rowSz);
        MPI_Request reqs[4]; int q=0;
        if(rank>0)        MPI_Irecv(recvT.data(),halo*rowSz,MPI_FLOAT,rank-1,0,MPI_COMM_WORLD,&reqs[q++]);
        if(rank<size-1)   MPI_Irecv(recvB.data(),halo*rowSz,MPI_FLOAT,rank+1,1,MPI_COMM_WORLD,&reqs[q++]);
        if(rank>0)        MPI_Isend(myIn.data(),halo*rowSz,MPI_FLOAT,rank-1,1,MPI_COMM_WORLD,&reqs[q++]);
        if(rank<size-1)   MPI_Isend(myIn.data()+(myIn.size()-halo*rowSz),halo*rowSz,MPI_FLOAT,
                                    rank+1,0,MPI_COMM_WORLD,&reqs[q++]);
        MPI_Waitall(q,reqs,MPI_STATUSES_IGNORE);
        if(rank>0)        myIn.insert(myIn.begin(),recvT.begin(),recvT.end());
        if(rank<size-1)   myIn.insert(myIn.end()  ,recvB.begin(),recvB.end());
    }

    /* ------------ copy padded slice to GPU --------------------- */
    float *d_in=nullptr; CUDA_CHECK(cudaMalloc(&d_in,myIn.size()*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in,myIn.data(),myIn.size()*sizeof(float),cudaMemcpyHostToDevice));

    /* ------------ temporary output buffer ---------------------- */
    // — Conv1 output dims —
    int Hc1 = convOutDim((int)myIn.size()/rowSz, p1.F, p1.P, p1.S);
    int Wc1 = convOutDim(W,                     p1.F, p1.P, p1.S);
    // — Pool1 output dims —
    int Hp1 = poolOutDim(Hc1, p1.F_pool, p1.S_pool);
    int Wp1 = poolOutDim(Wc1, p1.F_pool, p1.S_pool);
    // — Conv2 output dims —
    int Hc2 = convOutDim(Hp1, p2.F, p2.P, p2.S);
    int Wc2 = convOutDim(Wp1, p2.F, p2.P, p2.S);
    // — Pool2 output dims —
    int Hp2 = poolOutDim(Hc2, p2.F_pool, p2.S_pool);
    int Wp2 = poolOutDim(Wc2, p2.F_pool, p2.S_pool);

    std::vector<float> tileOut((size_t)Hp2 * Wp2 * p2.K);

    float* d_out; CUDA_CHECK(cudaMalloc(&d_out,tileOut.size()*sizeof(float)));

    /* ------------ single-GPU forward on the tile --------------- */
    alexnetTileForwardCUDA(d_in,p1,p2,(int)myIn.size()/rowSz,W,C,d_out);
    CUDA_CHECK(cudaMemcpy(tileOut.data(),d_out,
                          tileOut.size()*sizeof(float),cudaMemcpyDeviceToHost));

    /* ------------ trim halo-only rows -------------------------- */
    int trim1=poolOutDim(halo,p1.F_pool,p1.S_pool);
    int trim2=poolOutDim(halo,p2.F_pool,p2.S_pool);
    int trim = trim1 + trim2;
    int start=(rank?trim:0), stop=Hp2-(rank<size-1?trim:0);
    std::vector<float> local(tileOut.begin()+start*Wp2*p2.K,
                             tileOut.begin()+stop *Wp2*p2.K);

    /* ------------ gather back ---------------------------------- */
    int sendN=(int)local.size();
    std::vector<int> recN(size),dispG(size);
    MPI_Gather(&sendN,1,MPI_INT,recN.data(),1,MPI_INT,0,MPI_COMM_WORLD);
    if(rank==0){dispG[0]=0;for(int i=1;i<size;++i)dispG[i]=dispG[i-1]+recN[i-1];
                full.resize(dispG.back()+recN.back());}
    MPI_Gatherv(local.data(),sendN,MPI_FLOAT,
                full.data(),recN.data(),dispG.data(),MPI_FLOAT,0,MPI_COMM_WORLD);

    if(rank==0){
        std::cout<<"shape "<<full.size()/(Wp2*p2.K)<<"x"<<Wp2<<"x"<<p2.K<<"\n"
                 <<"sample "<<full[0]<<" "<<full[1]<<" "<<full[2]<<" "
                 <<full[3]<<" "<<full[4]<<"\n";
    }
    cudaFree(d_in); cudaFree(d_out);
}
/*----------------------------------------------------------------*/
int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);
    LayerParams b1,b2;
    b1.K=96 ; b1.F=11; b1.S=4; b1.P=0; b1.F_pool=3; b1.S_pool=2;
    b2.K=256; b2.F=5 ; b2.S=1; b2.P=2; b2.F_pool=3; b2.S_pool=2;
    b2.N_lrn=5; b2.alpha=1e-4f; b2.beta=0.75f; b2.k_lrn=2.f;

    const int H=227,W=227,C=3;
    MPI_Barrier(MPI_COMM_WORLD);
    auto t0=std::chrono::high_resolution_clock::now();
    runHybridMPI(H,W,C,b1,b2);
    MPI_Barrier(MPI_COMM_WORLD);
    if(int rank;!MPI_Comm_rank(MPI_COMM_WORLD,&rank)&&rank==0){
        auto t1=std::chrono::high_resolution_clock::now();
        std::cout<<"total runtime "
                 <<std::chrono::duration<double,std::milli>(t1-t0).count()
                 <<" ms\n";
    }
    MPI_Finalize();
    return 0;
}

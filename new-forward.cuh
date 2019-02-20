
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define CUDA_MAX_NUM_THREADS 1024
#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define TILE_SIZE 16.0
#define TILE_SIZE_ 16
//__constant__ float kernel[]
//__constant__ float deviceKernel[14112];


__global__ void gemm_Kernel1(const float * __restrict__ X_unrolled, float * __restrict__ y , const float * __restrict__ kernel)
{
    __shared__ float input1[TILE_SIZE_ * TILE_SIZE_];
    __shared__ float input2[TILE_SIZE_ * TILE_SIZE_];
    __shared__ float localkernel1[TILE_SIZE_ * TILE_SIZE_];
    __shared__ float localkernel2[TILE_SIZE_ * TILE_SIZE_];
    float *input = input1, *input_ = input2;
    float *localkernel = localkernel1, *localkernel_ = localkernel2;
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y, b = blockIdx.z;
    int row = by * TILE_SIZE_ + ty, col = bx * TILE_SIZE_ + tx;
    int h_out = col / 66, w_out = col % 66;
    float result = 0;

    #pragma unroll
    // 49 / TILEsize ceil
    for(int ph = 0;ph < 3;ph++)
    {
        int c = (ph * TILE_SIZE_ + ty) / 49;
        int s = (ph * TILE_SIZE_ + ty) % 49;
        int h = s / 7, w = s % 7;

        
        if(ph * TILE_SIZE_ + ty < 49 && col < 4356)
            input[ty * TILE_SIZE_ + tx] = X_unrolled[b * 5184 + c * 5184 + (h_out + h) * 72 + (w_out + w)];
        else
            input[ty * TILE_SIZE_ + tx] = 0;
        if(row < 12 && ph * TILE_SIZE_ + tx < 49)
            localkernel[ty * TILE_SIZE_ + tx] = kernel[row * 49 + ph * TILE_SIZE_ + tx];
        else
            localkernel[ty * TILE_SIZE_ + tx] = 0;
        
        __syncthreads();
        #pragma unroll
        for(int i = 0;i < TILE_SIZE_;i++)
            result += localkernel[ty * TILE_SIZE_ + i] * input[i * TILE_SIZE_ + tx];
        float *tmp1 = input, *tmp2 = localkernel;
        input = input_;
        input_ = tmp1;
        localkernel = localkernel_;
        localkernel_ = tmp2;

    }
    if(row < 12 && col < 4356)
        y[b * 52272 + row * 4356 + col] = result;
    
}
__global__ void gemm_Kernel2(const float * __restrict__ X_unrolled, float * __restrict__ y , const float * __restrict__ kernel)
{
    __shared__ float input1[TILE_SIZE_ * TILE_SIZE_];
    __shared__ float input2[TILE_SIZE_ * TILE_SIZE_];
    __shared__ float localkernel1[TILE_SIZE_ * TILE_SIZE_];
    __shared__ float localkernel2[TILE_SIZE_ * TILE_SIZE_];
    float *input = input1, *input_ = input2;
    float *localkernel = localkernel1, *localkernel_ = localkernel2;

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y, b = blockIdx.z;
    int row = by * TILE_SIZE + ty, col = bx * TILE_SIZE + tx;
    int h_out = col / 27, w_out = col % 27;
    //int H_unroll = 588, W_unroll = 729;
    float result = 0;

    #pragma unroll
    //588 / tilesize ceil
    for(int ph = 0;ph < 37;ph++)
    {
        int c = (ph * TILE_SIZE_ + ty) / 49;
        int s = (ph * TILE_SIZE_ + ty) % 49;
        int h = s / 7, w = s % 7;

        
        if(ph * TILE_SIZE + ty < 588 && col < 729)
            input[ty * TILE_SIZE_ + tx] = X_unrolled[b * 13068 + c * 1089 + (h_out + h) * 33 + (w_out + w)];
        else
            input[ty * TILE_SIZE_ + tx] = 0;
        if(row < 24 && ph * TILE_SIZE + tx < 588)
            localkernel[ty * TILE_SIZE_ + tx] = kernel[row * 588 + ph * TILE_SIZE_ + tx];
        else
            localkernel[ty * TILE_SIZE_ + tx] = 0;
        
        __syncthreads();
        #pragma unroll
        for(int i = 0;i < TILE_SIZE;i++)
            result += localkernel[ty * TILE_SIZE_ + i] * input[i * TILE_SIZE_ + tx];
        float *tmp1 = input, *tmp2 = localkernel;
        input = input_;
        input_ = tmp1;
        localkernel = localkernel_;
        localkernel_ = tmp2;

    }
    if(row < 24 && col < 729)
        y[b * 17496 + row * 729 + col] = result;
    
}
/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    
    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[2];
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_unroll = H_out * W_out;

    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
    dim3 dimGrid(ceil(W_unroll / TILE_SIZE), ceil(M / TILE_SIZE), B);
    if(C == 1)
        gemm_Kernel1<<<dimGrid, dimBlock, 0>>>(x.dptr_, y.dptr_, w.dptr_);
    else
        gemm_Kernel2<<<dimGrid, dimBlock, 0>>>(x.dptr_, y.dptr_, w.dptr_);


}
#undef TILE_SIZE
/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
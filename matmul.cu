/* Inference for GGUF Qwen-3 models in pure CUDA */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


__global__ void matmul_kernel(float *xout, float *x, float *w, int n, int d, int chunk_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float shared_x[];
    
    // Load x into shared memory in chunks
    for (int offset = 0; offset < n; offset += chunk_size) {
        if (offset + tid < n) {
            shared_x[tid] = x[offset + tid];
        }
        __syncthreads();
        
        if (i < d) {
            float sum = 0.0f;
            int current_chunk_size = min(chunk_size, n - offset);
            
            // Vectorized loads and computation
            float4 *w_vec = (float4*)(w + i * n + offset);
            float4 *x_vec = (float4*)shared_x;
            
            int vec_ops = current_chunk_size / 4;
            for (int v = 0; v < vec_ops; v++) {
                float4 w4 = w_vec[v];
                float4 x4 = x_vec[v];
                sum += w4.x * x4.x + w4.y * x4.y + w4.z * x4.z + w4.w * x4.w;
            }
            
            // Handle remaining elements
            for (int j = vec_ops * 4; j < current_chunk_size; j++) {
                sum += w[i * n + offset + j] * shared_x[j];
            }
            
            if (offset == 0) xout[i] = sum;
            else xout[i] += sum;
        }
        __syncthreads();
    }
}

void matmul(float *xout, float *x, float *w, int n, int d, int block_size, int chunk_size, int shared_mem_size) {
    int grid_size = (d + block_size - 1) / block_size;
    matmul_kernel<<<grid_size, block_size, shared_mem_size>>>(xout, x, w, n, d, chunk_size);
}


int main(int argc, char *argv[]) {
    // Default parameters
    int n = 4096;  // input dimension
    int d = 4096;  // output dimension
    int warmup = 10;
    int iterations = 100;
    
    // Parse command line arguments
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) d = atoi(argv[2]);
    if (argc > 3) iterations = atoi(argv[3]);
    
    printf("Matmul autotuning: n=%d, d=%d, iterations=%d\n\n", n, d, iterations);
    
    // Allocate host memory
    float *h_x = (float*)malloc(n * sizeof(float));
    float *h_w = (float*)malloc(d * n * sizeof(float));
    float *h_xout = (float*)malloc(d * sizeof(float));
    
    // Initialize with random values
    for (int i = 0; i < n; i++) h_x[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < d * n; i++) h_w[i] = (float)rand() / RAND_MAX;
    
    // Allocate device memory
    float *d_x, *d_w, *d_xout;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_w, d * n * sizeof(float));
    cudaMalloc(&d_xout, d * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, d * n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Parameters to test
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int chunk_sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    int num_block_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);
    int num_chunk_sizes = sizeof(chunk_sizes) / sizeof(chunk_sizes[0]);
    
    float best_gflops = 0.0f;
    int best_block_size = 0;
    int best_chunk_size = 0;
    
    // Test each combination
    for (int bs_idx = 0; bs_idx < num_block_sizes; bs_idx++) {
        int block_size = block_sizes[bs_idx];
        
        for (int cs_idx = 0; cs_idx < num_chunk_sizes; cs_idx++) {
            int chunk_size = chunk_sizes[cs_idx];
            int shared_mem_size = chunk_size * sizeof(float);
            
            // Skip if shared memory is too large
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            if (shared_mem_size > prop.sharedMemPerBlock) {
                continue;
            }
            
            // Warmup
            for (int i = 0; i < warmup; i++) {
                matmul(d_xout, d_x, d_w, n, d, block_size, chunk_size, shared_mem_size);
            }
            cudaDeviceSynchronize();
            
            // Check for errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("block_size=%4d, chunk_size=%4d: SKIP (error: %s)\n", 
                       block_size, chunk_size, cudaGetErrorString(err));
                continue;
            }
            
            // Benchmark
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            cudaEventRecord(start);
            for (int i = 0; i < iterations; i++) {
                matmul(d_xout, d_x, d_w, n, d, block_size, chunk_size, shared_mem_size);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            
            // Calculate metrics
            float avg_time = milliseconds / iterations;
            float gflops = (2.0f * n * d / 1e9) / (avg_time / 1000.0f);
            
            printf("block_size=%4d, chunk_size=%4d, shared_mem=%5d KB: %.4f ms, %.2f GFLOPS\n", 
                   block_size, chunk_size, shared_mem_size / 1024, avg_time, gflops);
            
            if (gflops > best_gflops) {
                best_gflops = gflops;
                best_block_size = block_size;
                best_chunk_size = chunk_size;
            }
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }
    
    printf("\nBest configuration: block_size=%d, chunk_size=%d (%.2f GFLOPS)\n", 
           best_block_size, best_chunk_size, best_gflops);
    
    // Copy result back
    cudaMemcpy(h_xout, d_xout, d * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    free(h_x);
    free(h_w);
    free(h_xout);
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_xout);
    
    return 0;
}

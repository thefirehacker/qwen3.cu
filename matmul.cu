/* Inference for GGUF Qwen-3 models in pure CUDA */

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256


__global__ void matmul_kernel(float *xout, float *x, float *w, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float shared_x[];
    
    // Load x into shared memory in chunks
    for (int offset = 0; offset < n; offset += blockDim.x) {
        if (offset + tid < n) {
            shared_x[tid] = x[offset + tid];
        }
        __syncthreads();
        
        if (i < d) {
            float sum = 0.0f;
            int chunk_size = min(blockDim.x, n - offset);
            
            // Vectorized loads and computation
            float4 *w_vec = (float4*)(w + i * n + offset);
            float4 *x_vec = (float4*)shared_x;
            
            int vec_ops = chunk_size / 4;
            for (int v = 0; v < vec_ops; v++) {
                float4 w4 = w_vec[v];
                float4 x4 = x_vec[v];
                sum += w4.x * x4.x + w4.y * x4.y + w4.z * x4.z + w4.w * x4.w;
            }
            
            // Handle remaining elements
            for (int j = vec_ops * 4; j < chunk_size; j++) {
                sum += w[i * n + offset + j] * shared_x[j];
            }
            
            if (offset == 0) xout[i] = sum;
            else xout[i] += sum;
        }
        __syncthreads();
    }
}

void matmul(float *xout, float *x, float *w, int n, int d, int b_size) {
    int block_size = b_size;
    int grid_size = (d + block_size - 1) / block_size;
    int shared_mem = block_size * sizeof(float);
    matmul_kernel<<<grid_size, block_size, shared_mem>>>(xout, x, w, n, d);
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
    
    // Block sizes to test
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int num_block_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);
    
    float best_gflops = 0.0f;
    int best_block_size = 0;
    
    // Test each block size
    for (int bs_idx = 0; bs_idx < num_block_sizes; bs_idx++) {
        int block_size = block_sizes[bs_idx];
        
        // Warmup
        for (int i = 0; i < warmup; i++) {
            matmul(d_xout, d_x, d_w, n, d, block_size);
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            matmul(d_xout, d_x, d_w, n, d, block_size);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        // Calculate metrics
        float avg_time = milliseconds / iterations;
        float gflops = (2.0f * n * d / 1e9) / (avg_time / 1000.0f);
        
        printf("block_size=%4d: %.4f ms, %.2f GFLOPS\n", block_size, avg_time, gflops);
        
        if (gflops > best_gflops) {
            best_gflops = gflops;
            best_block_size = block_size;
        }
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    printf("\nBest configuration: block_size=%d (%.2f GFLOPS)\n", best_block_size, best_gflops);
    
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
#ifndef ODD_EVEN_CU
#define ODD_EVEN_CU

__global__ void GPU_odd_even_kernel(int* vec, int size, int* done) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int og_tid = tid;
    bool local_done = false;
    bool local_change;
    int start = og_tid * 2;

    atomicAdd(done, 1);

    while (*done > 0) {
        tid = start;
        local_change = false;

        while (tid < size - 1) {
            if (vec[tid] > vec[tid + 1]) {
                int temp     = vec[tid];
                vec[tid]     = vec[tid + 1];
                vec[tid + 1] = temp;
                local_change = true;
        
            }
            tid += blockDim.x * gridDim.x * 2;    
        }
        __syncthreads();

        tid = start + 1;
        while (tid < size - 1) {

            if (vec[tid] > vec[tid + 1]) {
                int temp     = vec[tid];
                vec[tid]     = vec[tid + 1];
                vec[tid + 1] = temp;
                local_change = true;
            }
            
            tid += blockDim.x * gridDim.x * 2;   
        }
        __syncthreads();

        if (local_change) {
            if (local_done == true) {
                local_done = false;
                atomicAdd(done, 1);
            }
        } else {
            if (local_done == false) {
                local_done = true;
                atomicAdd(done, -1);
            }
        }
        
    }
}

void GPU_odd_even_sort(int* vec, int size) {
    int* deviceVec;
    H_ERR(cudaMalloc((void**) &deviceVec, sizeof(int) * size));

    H_ERR(cudaMemcpy(deviceVec, vec, sizeof(int) * size, cudaMemcpyHostToDevice));

    GPU_odd_even_kernel <<<128, 128>>> (deviceVec, size);

    H_ERR( cudaDeviceSynchronize());

    H_ERR(cudaMemcpy(vec, deviceVec, sizeof(int) * size, cudaMemcpyDeviceToHost));

    
}

#endif  // ODD_EVEN_CU
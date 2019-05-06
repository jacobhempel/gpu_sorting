#ifndef ODD_EVEN_CU
#define ODD_EVEN_CU

#include "util.h"
#include <time.h>


bool is_sorted(int* array, int size);
void shuffle(int* array, int size);
void print_array(int* array, int size);

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
        __syncthreads();
    }
}

__device__ void swap(int* a, int* b) { 
    int t = *a; 
    *a = *b; 
    *b = t; 
} 
  
__device__ int partition(int arr[], int l, int h) { 
    int x = arr[h]; 
    int i = (l - 1); 
              
    for (int j = l; j <= h - 1; j++) { 
         if (arr[j] <= x) { 
             i++; 
             swap(&arr[i], &arr[j]); 
         } 
    } 
    swap(&arr[i + 1], &arr[h]); 
    return (i + 1); 
} 
  
__device__ void quickSort(int* arr, int* stack, int l, int h) { 
    
    int top = l - 1; 
    int start = l; 
    stack[++top] = l; 
    stack[++top] = h; 
                      
    while (top >= start) { 
        h = stack[top--]; 
        l = stack[top--]; 
        int p = partition(arr, l, h); 
                                                      
        if (p - 1 > l) { 
            stack[++top] = l; 
            stack[++top] = p - 1; 
        } 
                                                              
        if (p + 1 < h) { 
            stack[++top] = p + 1; 
            stack[++top] = h; 
        } 
    }
} 



__global__ void samplesort_kernel(int* array, int* blocks, int* stack, int* result_index, int* splitters, int* worker, int size, int size_of_splitters, int blockSize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int og_tid = tid;
    
    int left;
    int right;

    if (tid == 0) {
        left = -2100000000;
        right = splitters[tid];
    } 
    else if (tid != 0 && tid < size_of_splitters - 1) {
        left =  splitters[tid - 1];
        right = splitters[tid];
    }
    else if (tid == size_of_splitters - 1) {
        left = splitters[tid - 1];
        right = 2100000000;
    }
    else {
        return;
    }
    
    // printf("%d grabbing data\n", og_tid);

    int j = tid * blockSize;
    int start = j;
    int len = 0;

    for (int i = 0; i < size; i++) {
        if (array[i] < right && array[i] >= left) {
            blocks[j] = array[i];
            j++;
            len++;
        }
    }

    // printf("%d starting sorting @ %d + %d\n", og_tid, start, j - 1);
    quickSort(blocks, stack, start, j - 1);
    // printf("%d done sorting\n", og_tid);

    splitters[tid] = len;
    
    // printf("%d done\n", og_tid);
}


void GPU_odd_even_sort(int* vec, int size) {
    int* deviceVec;
    int* done;
    int zero = 0;
    H_ERR(cudaMalloc((void**) &deviceVec, sizeof(int) * size));
    H_ERR(cudaMalloc((void**) &done, sizeof(int)));

    H_ERR(cudaMemcpy(deviceVec, vec, sizeof(int) * size, cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(done, &zero, sizeof(int), cudaMemcpyHostToDevice));

    GPU_odd_even_kernel <<<128, 128>>> (deviceVec, size, done);

    H_ERR( cudaDeviceSynchronize());
    
    H_ERR(cudaMemcpy(vec, deviceVec, sizeof(int) * size, cudaMemcpyDeviceToHost));

    
}

bool GPU_sample_sort(int* array, int size) {
    const int GPU_THREADS = 256; 
    int samplesize = 4096;

    int* devArray;
    int* worker;
    int* devSplitters;
    int* devResultIndex;
    int* devBlocks;
    int blockSize = 4 * size / GPU_THREADS; 
    int blocksSize = 4 * size;
    int* devStack;
    int splitters[GPU_THREADS];
    int* samples = NULL;
    samples = (int*) malloc (sizeof(int) * samplesize);

    int zero = 0;

    // printf("made my variables!!\n");

    for (int i = 0; i < samplesize; i++) {
        int j = rand() % size;
        samples[i] = array[j];
        // printf("grabbing sample from: %d\n", j);   
    }

    do {
        GPU_odd_even_sort(samples, samplesize); 
    } while (!is_sorted(samples, samplesize)); 
    // print_array(samples, samplesize); 
    
    int erval = samplesize / GPU_THREADS;
    int j = 0;
    for (int i = erval - 1; i < samplesize; i += erval) {
        splitters[j] = samples[i];
        j++;
    }

    // print_array(splitters, GPU_THREADS - 1);

    H_ERR(cudaMalloc((void**) &devArray, sizeof(int) * size));
    H_ERR(cudaMalloc((void**) &worker, sizeof(int)));
    H_ERR(cudaMalloc((void**) &devSplitters, sizeof(int) * GPU_THREADS));
    H_ERR(cudaMalloc((void**) &devResultIndex, sizeof(int)));
    H_ERR(cudaMalloc((void**) &devBlocks, sizeof(int) * blocksSize));
    H_ERR(cudaMalloc((void**) &devStack, sizeof(int) * blocksSize));

    H_ERR(cudaMemcpy(devArray, array, sizeof(int) * size, cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(worker, &zero, sizeof(int), cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(devSplitters, splitters, sizeof(splitters), cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(devResultIndex, &zero, sizeof(int), cudaMemcpyHostToDevice));

    // print_array(array, size);

    samplesort_kernel <<< 256, 256 >>> (devArray, devBlocks, devStack, devResultIndex, devSplitters, worker, size, GPU_THREADS, blockSize);
    
    H_ERR(cudaDeviceSynchronize());
    
    int* blocks = (int* ) malloc (sizeof(int) * blocksSize);

    printf("copying results back\n");
    H_ERR(cudaMemcpy(blocks, devBlocks, sizeof(int) * blocksSize, cudaMemcpyDeviceToHost));
    H_ERR(cudaMemcpy(splitters, devSplitters, sizeof(int) * GPU_THREADS, cudaMemcpyDeviceToHost));
    
    int k = 0;
    for (int i = 0; i < GPU_THREADS; i++) {
        int len = splitters[i];
        int BI = blockSize * i; 
        for (int j = BI; j < BI + len; j++) {
            array[k++] = blocks[j];
        }
    }
    printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    
    cudaFree(devArray);
    cudaFree(worker);
    cudaFree(devSplitters);
    cudaFree(devResultIndex);
    cudaFree(devBlocks);
    cudaFree(devStack);
   
    

    bool done = is_sorted(array, size);
    if (done) {
        // print_array(result, size);
        return true;
    } else {
        // print_array(array, size);
        // print_array(splitters, GPU_THREADS);
        return false;
    }
}

void do_GPU_oddeven_sort(int* array, int size) {
    shuffle(array, size);
    print_array(array, size);
    clock_t t1 = clock();
    GPU_odd_even_sort(array, size);
    clock_t t2 = clock();
    if (is_sorted(array, size)) {
        printf("gpu odd even sort took %f seconds\n", ((double) t2 - t1) / CLOCKS_PER_SEC);
    } else {
        printf("GPU ODDEVEN SORT FAILED TO PRODUCE A SORTED LIST\n");
        print_array(array, size);
    }
}

void do_GPU_samplesort(int* array, int size) {
    shuffle(array, size);
    // print_array(array, size);
    clock_t t1 = clock();
    bool sorted = GPU_sample_sort(array, size);
    clock_t t2 = clock();
    if (sorted) {
        printf("gpu samplesort took %f seconds\n", ((double) t2 - t1) / CLOCKS_PER_SEC);
    } else {
        printf("GPU SAMPLESORT FAILED TO PRODUCE A SORTED LIST\n");
        // print_array(array, size);
    }
}

int main(int argc, char* argv[]) {
    int size = 10000000; 
    srand(time(0));

    for (int i = 0; i < 10; i++) { 

        int* array = (int*) malloc (sizeof(int) * size);
        for (int i = 0; i < size; i++) {
            array[i] = i;
        }
    
        do_GPU_samplesort(array, size);
        free(array);
    }
}


bool is_sorted(int* array, int size) {
    for (int i = 0; i < size - 1; i++) {
        if (array[i] > array[i + 1]) {
            return false;
        }
    }
    return true;
}

void shuffle(int* array, int size) {
    for (int i = 0; i < size; i++) {
        int temp = array[i];
        int swap = rand() % size;
        array[i] = array[swap];
        array[swap] = temp;
    }
}

void print_array(int* array, int size) {
    for (int i = 0; i < size; i++) { 
        printf("array[%d] = %d\n", i, array[i]); 
    }   
}

#endif  // ODD_EVEN_CU

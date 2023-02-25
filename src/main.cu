#include <stdio.h>
#include <stdlib.h>
// #include <time.h>
#include <chrono>

extern "C"{
#include "csx.h"
}

// #define TIME_OP(NAME, OP)\
//   clock_gettime(CLOCK_MONOTONIC, &T_START); \
//   OP; \
//   clock_gettime(CLOCK_MONOTONIC, &T_END); \
//   {double t_elapsed = (double)(T_END.tv_sec - T_START.tv_sec) * 1000 + (double)(T_END.tv_nsec - T_START.tv_nsec) / 1000000; \
//   printf("%s took %f ms\n", NAME,  t_elapsed);};

#define TIME_OP(NAME, OP) \
      T_START = std::chrono::high_resolution_clock::now(); \
      OP; \
      T_END = std::chrono::high_resolution_clock::now(); \
      printf("%s took %f ms\n", NAME,  (double)std::chrono::duration_cast<std::chrono::microseconds>(T_END-T_START).count()/1000.0);
      // std::cout << name << " took " << std::chrono::duration_cast<std::chrono::milliseconds>(T_END-T_START).count() << " ms" << std::endl;


#define TIMER_START \
  clock_gettime(CLOCK_MONOTONIC, &T_START);

#define TIMER_STOP(NAME) \
  clock_gettime(CLOCK_MONOTONIC, &T_END); \
  {double t_elapsed = (double)(T_END.tv_sec - T_START.tv_sec) * 1000 + (double)(T_END.tv_nsec - T_START.tv_nsec) / 1000000; \
  printf("%s took %f ms\n", NAME,  t_elapsed);}



__global__
void kernel_spmv(size_t n_vertices, size_t* A_com, size_t* A_unc, size_t *x, size_t *y) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < n_vertices){
    y[idx] = 0;

    for (int k = A_com[idx]; k < A_com[idx + 1]; k++) {
      y[idx] += x[A_unc[k]];
    } 
  }
}


__global__
void kernel_c3(size_t n_vertices, size_t* A_com, size_t* A_unc, size_t *c3) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_vertices) {
    const int i_nb_start = A_com[i];
    const int i_nb_end = A_com[i + 1];

    for (int i_nb_idx = i_nb_start; i_nb_idx < i_nb_end; i_nb_idx++) {
      int j = A_unc[i_nb_idx];

      if(i<=j) break;

      const int j_nb_start = A_com[j];
      const int j_nb_end = A_com[j + 1];

      int _i_nb_idx = i_nb_start;
      int _j_nb_idx = j_nb_start;

      while (_i_nb_idx < i_nb_end && _j_nb_idx < j_nb_end)
      {
        if ((A_unc[_i_nb_idx] > i) || (A_unc[_j_nb_idx] > j)){
            break;
        }
        else if (A_unc[_i_nb_idx] == A_unc[_j_nb_idx])
        {
          c3[j]++;
          c3[i]++;
          c3[A_unc[_i_nb_idx]]++;
          _i_nb_idx++;
          _j_nb_idx++;
        }
        else if (A_unc[_i_nb_idx] < A_unc[_j_nb_idx])
        {
          _i_nb_idx++;
        }
        else
        {
          _j_nb_idx++;
        }
      }
    }
  }
}


__global__
void kernel_s0(size_t n_vertices, size_t* s0){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_vertices) {
    (s0)[idx] = 1;
  }
}
__global__
void kernel_s1(size_t n_vertices, size_t* A_com, size_t* s1){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_vertices) {
    s1[idx] = A_com[idx + 1] - A_com[idx];
  }
}

__global__
void kernel_s2(size_t n_vertices, size_t* s1, size_t* s2){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_vertices) {
    s2[idx] -= s1[idx];
  }
}

__global__
void kernel_s3(size_t n_vertices, size_t* s1, size_t* s3){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_vertices) {
    s3[idx] = (s1[idx] * (s1[idx] - 1)) / 2;
  }
}

__global__
void kernel_s4(size_t n_vertices, size_t* s2, size_t* s3, size_t* s4){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_vertices) {
     s2[idx] -= 2 * s4[idx];
     s3[idx] -= s4[idx];
  }
}



void fglt(csx h_A){
  // struct timespec T_START, T_END;
  std::chrono::high_resolution_clock::time_point T_START, T_END;

  // Allocate device vectors
  size_t *d_d0, *d_d1, *d_d2, *d_d3, *d_d4;
  cudaMalloc(&d_d0, h_A->v * sizeof(size_t));
  cudaMalloc(&d_d1, h_A->v * sizeof(size_t));
  cudaMalloc(&d_d2, h_A->v * sizeof(size_t));
  cudaMalloc(&d_d3, h_A->v * sizeof(size_t));
  cudaMalloc(&d_d4, h_A->v * sizeof(size_t));

  // Allocate host vectors
  size_t *h_d0, *h_d1, *h_d2, *h_d3, *h_d4;
  h_d0 = (size_t*)calloc(h_A->v, sizeof(size_t));
  h_d1 = (size_t*)calloc(h_A->v, sizeof(size_t));
  h_d2 = (size_t*)calloc(h_A->v, sizeof(size_t));
  h_d3 = (size_t*)calloc(h_A->v, sizeof(size_t));
  h_d4 = (size_t*)calloc(h_A->v, sizeof(size_t));
 
  size_t* d_A_com, *d_A_unc;
  TIME_OP("Moving the matrix to the device",
    // Send A to Device
    cudaMalloc(&d_A_com, (h_A->v + 1)* sizeof(size_t));
    cudaMalloc(&d_A_unc, h_A->e * sizeof(size_t));
    cudaMemcpy(d_A_com, h_A->com, (h_A->v + 1)* sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_unc, h_A->unc, h_A->e * sizeof(size_t), cudaMemcpyHostToDevice);
  )


  // Run calculations on device
  int blockSize = 1024;
  int numBlocks = (h_A->v + blockSize - 1) / blockSize;

  TIME_OP("d0",
  (kernel_s0   <<<numBlocks, blockSize>>>(h_A->v, d_d0));
  cudaDeviceSynchronize();
  );

  TIME_OP("d1",
  (kernel_s1   <<<numBlocks, blockSize>>>(h_A->v, d_A_com , d_d1));
  cudaDeviceSynchronize();
  );

  TIME_OP("d2",
  (kernel_spmv <<<numBlocks, blockSize>>>(h_A->v, d_A_com, d_A_unc, d_d1, d_d2));
  (kernel_s2   <<<numBlocks, blockSize>>>(h_A->v, d_d1, d_d2));
  cudaDeviceSynchronize();
  );

  TIME_OP("d3",
  (kernel_s3   <<<numBlocks, blockSize>>>(h_A->v, d_d1, d_d3));
  cudaDeviceSynchronize();
  );
  
  TIME_OP("c3",
  (kernel_c3 <<<numBlocks, blockSize>>>(h_A->v, d_A_com, d_A_unc, d_d4)); 
  cudaDeviceSynchronize();
  );
  
  TIME_OP("d4",
  (kernel_s4   <<<numBlocks, blockSize>>>(h_A->v, d_d2, d_d3, d_d4));
  cudaDeviceSynchronize();
  );


  // Transfer results from device to host
  TIME_OP("Moving the results back to the host",
  cudaMemcpy(h_d0, d_d0, (h_A->v)*sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_d1, d_d1, (h_A->v)*sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_d2, d_d2, (h_A->v)*sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_d3, d_d3, (h_A->v)*sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_d4, d_d4, (h_A->v)*sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  );
  
  // // Validate Result
  // size_t s0=0, s1=0, s2=0, s3=0, s4=0;
  // for(int i=0; i<h_A->v; i++){
  //   s0 += h_d0[i];
  //   s1 += h_d1[i];
  //   s2 += h_d2[i];
  //   s3 += h_d3[i];
  //   s4 += h_d4[i];
  // }
  // printf("s0:%lu\ns1:%lu\ns2:%lu\ns3:%lu\ns4:%lu\n", s0, s1, s2, s3, s4);


  // Free device memory
  cudaFree(d_d0);
  cudaFree(d_d1);
  cudaFree(d_d2);
  cudaFree(d_d3);
  cudaFree(d_d4);
  cudaFree(d_A_com);
  cudaFree(d_A_unc);

  // Free host memory
  free(h_d0);
  free(h_d1);
  free(h_d2);
  free(h_d3);
  free(h_d4);
}



int main(int argc, char *argv[]) {

  // struct timespec T_START, T_END;
  std::chrono::high_resolution_clock::time_point T_START, T_END;

  if (argc < 2) {
    fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
    exit(1);
  }

  // Read mtx file
  TIME_OP("Loading the file",
    csx h_A = csc_from_file(argv[1]);
  )

  TIME_OP("The whole fglt",   
  fglt(h_A);
  );
  csx_free(h_A);

  return 0;
}

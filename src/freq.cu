extern "C" { 
#include "freq.h"
}

void c3(csx A, size_t *c3);

freq freq_new(size_t number_of_vertices) {
  freq f = (freq)malloc(sizeof(struct FREQ));

  f->v = number_of_vertices;

  f->s0 = (size_t*)calloc(f->v, sizeof(size_t));
  f->s1 = (size_t*)calloc(f->v, sizeof(size_t));
  f->s2 = (size_t*)calloc(f->v, sizeof(size_t));
  f->s3 = (size_t*)calloc(f->v, sizeof(size_t));
  f->s4 = (size_t*)calloc(f->v, sizeof(size_t));

  return f;
}

void freq_free(freq f) {
  free(f->s0);
  free(f->s1);
  free(f->s2);
  free(f->s3);
  free(f->s4);
  free(f);
}


freq freq_d_new(size_t number_of_vertices) {
  // f_d and it's referenced addresses exist in host space
  // However, f->s0, f->s1 are pointers to device memory
  freq f_d = (freq)malloc(sizeof(struct FREQ));

  f_d->v = number_of_vertices;

  cudaMalloc(&f_d->s0, f_d->v * sizeof(size_t));
  cudaMalloc(&f_d->s1, f_d->v * sizeof(size_t));
  cudaMalloc(&f_d->s2, f_d->v * sizeof(size_t));
  cudaMalloc(&f_d->s3, f_d->v * sizeof(size_t));
  cudaMalloc(&f_d->s4, f_d->v * sizeof(size_t));
  
  return f_d;
}

void freq_d_free(freq f) {
  cudaFree(f->s0);
  cudaFree(f->s1);
  cudaFree(f->s2);
  cudaFree(f->s3);
  cudaFree(f->s4);
  free(f);
}

void freq_device_to_host(freq f, freq f_d){
  // Copy contents of f_d (which exists on Device) to freq f (which exists on Host)

  cudaMemcpy(f->s0, f_d->s0, (f->v)*sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(f->s1, f_d->s1, (f->v)*sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(f->s2, f_d->s2, (f->v)*sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(f->s3, f_d->s3, (f->v)*sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(f->s4, f_d->s4, (f->v)*sizeof(size_t), cudaMemcpyDeviceToHost);

}

__global__
void kernel_spmv(size_t n_vertices, size_t* A_com, size_t* A_unc, size_t *x, size_t *y) {
  /* y = Ax 
  Each GPU thread calculates one element of the result vector */
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
  int j, k, l, lb, up, clb, cup, llb;
    lb = A_com[i];
    up = A_com[i + 1];
    for (j = lb; j < up; j++) {
      clb = A_com[A_unc[j]];
      cup = A_com[A_unc[j] + 1];
      llb = lb;
      for (k = clb; k < cup; k++) {
        for (l = llb; l < up; l++) {
          if (A_unc[k] == A_unc[l]) {
            c3[i]++;
            llb = l + 1;
            break;
          } else if (A_unc[k] < A_unc[l]) {
            llb = l;
            break;
          } else {
            llb = l + 1;
          }
        }
      }
    }
    c3[i] /= 2;
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

freq freq_calc(csx A) {

  freq f   = freq_new(A->v);
  freq f_d = freq_d_new(A->v);

  // Contents of A are sent to Device
  size_t* A_com_d, *A_unc_d;
  cudaMalloc(&A_com_d,        (A->v + 1)* sizeof(size_t));
  cudaMemcpy(A_com_d, A->com, (A->v + 1)* sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMalloc(&A_unc_d,        A->e * sizeof(size_t));
  cudaMemcpy(A_unc_d, A->unc, A->e * sizeof(size_t), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = (A->v + blockSize - 1) / blockSize;

  kernel_s0   <<<numBlocks, blockSize>>>(A->v, f_d->s0);
  kernel_s1   <<<numBlocks, blockSize>>>(A->v, A_com_d , f_d->s1);
  kernel_spmv <<<numBlocks, blockSize>>>(A->v, A_com_d, A_unc_d, f_d->s1, f_d->s2);
  kernel_s2   <<<numBlocks, blockSize>>>(A->v, f_d->s1, f_d->s2);
  kernel_s3   <<<numBlocks, blockSize>>>(A->v, f_d->s1, f_d->s3);
  
  kernel_c3   <<<numBlocks, blockSize>>>(A->v, A_com_d, A_unc_d, f_d->s4);
  kernel_s4   <<<numBlocks, blockSize>>>(A->v, f_d->s2, f_d->s3, f_d->s4);

  // // We could instead do c3 and s4 on the host
  // freq_device_to_host(f, f_d); //This is blocking! Probably not good for perf
  // c3(A, f->s4);
  // #pragma omp parallel for
  // for (size_t i = 0; i<A->v; i++){
  //   f->s2[i] -= 2 * f->s4[i];
  //   f->s3[i] -= f->s4[i];
  // }

  // // Or we could copy c3 to Device and do s4 calculation there
  // c3(A, f->s4);
  // cudaMemcpy(f_d->s4, f->s4, (A->v)*sizeof(size_t), cudaMemcpyHostToDevice);
  // kernel_s4   <<<numBlocks, blockSize>>>(A->v, f_d->s2, f_d->s3, f_d->s4);




  cudaDeviceSynchronize();
  freq_device_to_host(f, f_d);
  
  freq_d_free(f_d);
  return f;
}

void freq_print(freq f) {
  printf("  v   σ0  σ1  σ2  σ3  σ4\n");
  for (size_t v = 0; v < f->v; v++) {
    printf("%3zu: %3zu %3zu %3zu %3zu %3zu\n", v, f->s0[v], f->s1[v], f->s2[v],
           f->s3[v], f->s4[v]);
  }
};

/**
 * Caclucaltes y = Ax
 * \\
 * where A is a matrix in CSC/CSR format and x a dense vector
 */


/**
 * Caclucaltes c3 = (A ○ A^2)e / 2
 * \\
 * where A is a matrix in CSC/CSR format and e a vector with all elements equal
 * 1
 */
void c3(csx A, size_t *c3) {
  int j, k, l, lb, up, clb, cup, llb;
  #pragma omp parallel for
  for(int i = 0; i < A->v; i++) {
    lb = A->com[i];
    up = A->com[i + 1];
    for (j = lb; j < up; j++) {
      clb = A->com[A->unc[j]];
      cup = A->com[A->unc[j] + 1];
      llb = lb;
      for (k = clb; k < cup; k++) {
        for (l = llb; l < up; l++) {
          if (A->unc[k] == A->unc[l]) {
            c3[i]++;
            llb = l + 1;
            break;
          } else if (A->unc[k] < A->unc[l]) {
            llb = l;
            break;
          } else {
            llb = l + 1;
          }
        }
      }
    }
    c3[i] /= 2;
  }
}

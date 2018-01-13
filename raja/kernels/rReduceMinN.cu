#include <stdio.h>
#include <cuda_runtime.h>

// *****************************************************************************
#define BLOCKSIZE 128
#define CUDA_NB_THREADS_PER_BLOCK 128
#define PAD_DIV(nbytes, align) (((nbytes)+(align)-1)/(align))

// *****************************************************************************
__shared__ double shared_min_array[CUDA_NB_THREADS_PER_BLOCK];


// *****************************************************************************
__device__
void reduce_min_kernel(const int N, double *results, const double init){
  const unsigned int bid = blockIdx.x;
  const unsigned int tid = threadIdx.x;
  const unsigned int i = bid*blockDim.x+tid;
  int dualTid;
  // Le bloc dépose la valeure qu'il a
  //__syncthreads();
  shared_min_array[tid]=init;
  __syncthreads();
  for(int workers=blockDim.x>>1; workers>0; workers>>=1){
    // Seule la premiere moitié travaille
    if (tid >= workers) continue;
    dualTid = tid + workers;
    // On évite de piocher trop loin
    if (i >= N) continue;
    if (dualTid >= N) continue;
    if ((blockDim.x*bid + dualTid) >= N) continue;
    // Voici ceux qui travaillent
    //printf("\n#%03d/%d of bloc #%d <?= with #%d", tid, workers, blockIdx.x, dualTid);
    // On évite de taper dans d'autres blocs
    //if (dualTid >= blockDim.x) continue;
    // ALORS on peut réduire:
    {
      const double tmp = shared_min_array[dualTid];
      //printf("\n#%03d/%d of bloc #%d <?= with #%d: %0.15e vs %0.15e",tid, workers, blockIdx.x, dualTid,shared_min_array[tid],shared_min_array[dualTid]);
      if (tmp < shared_min_array[tid])
        shared_min_array[tid] = tmp;
    }
    __syncthreads();
  }
  __syncthreads();
  if (tid==0){
    results[bid]=shared_min_array[0];
    printf("\n\033[32mMin bloc #%d returned %0.15e\033[m", bid, results[bid]);
    __syncthreads();
  }
}

// *****************************************************************************
__global__
void cudaReduction_min(const int N,
                       double *results,
                       const double *data){ 
  const register int tid = blockDim.x*blockIdx.x + threadIdx.x;
  if (tid>=N) return;
  reduce_min_kernel(N,results,data[tid]);
}

// *****************************************************************************
void reduceMinN(int size, const double *d_idata, double *d_odata){
  //printf("\n\033[32m[reduceMinN] d_idata:\033[m");

  static double *t_idata=NULL;
  if (!t_idata){
    cudaMallocManaged(&t_idata, size*sizeof(double), cudaMemAttachGlobal);
  }

  for(int i=0;i<size;i+=1) {
    //t_idata[i]=(i+1)*0.0012345678;
    //for(int i=size-1;i>=0;i-=1) {
    //printf(" [%f]",t_idata[i]);
    //printf(" [%f]",d_idata[i]);
  }

  const dim3 dimJobBlock=dim3(BLOCKSIZE,1,1);
  const dim3 dimCellGrid=dim3(PAD_DIV(size,dimJobBlock.x),1,1);
  const int reduced_size=(size%CUDA_NB_THREADS_PER_BLOCK)==0?
    (size/CUDA_NB_THREADS_PER_BLOCK):(1+size/CUDA_NB_THREADS_PER_BLOCK);

  static double *results=NULL;
  if (!results){
    cudaMallocManaged(&results, reduced_size*sizeof(double), cudaMemAttachGlobal);
  }
 
  cudaReduction_min<<<dimCellGrid,dimJobBlock>>>(size,
                                                 results,
                                                 d_idata);
  cudaDeviceSynchronize();  
  d_odata[0]=results[0];
  for(int i=1;i<reduced_size;i+=1){
    //printf("\n\033[32m\t[reduceMin] reduced=%0.15e, reduce_results[%d/%d]=%0.15e\033[m",d_odata[0],i,results[i],reduced_size);
    d_odata[0]=(d_odata[0]<results[i])?d_odata[0]:results[i];
  }
  printf("\033[32m[reduceMin] d_odata[0]=%f\n\033[m",d_odata[0]);
}

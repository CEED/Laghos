#include <stdio.h>
#include <cuda_runtime.h>


// *****************************************************************************
#define BLOCKSIZE 128
#define CUDA_NB_THREADS_PER_BLOCK 128
#define PAD_DIV(nbytes, align) (((nbytes)+(align)-1)/(align))


// *****************************************************************************
__shared__ double shared_sum_array[CUDA_NB_THREADS_PER_BLOCK];


// *****************************************************************************
__device__
void reduce_sum_kernel(const int N, double *results, const double init1, const double init2){
  const unsigned int bid = blockIdx.x;
  const unsigned int tid = threadIdx.x;
  const unsigned int i = bid*blockDim.x+tid;
  int dualTid;
  // Le bloc dépose la valeure qu'il a
  //__syncthreads();
  shared_sum_array[tid]=init1*init2;
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
    //printf("\n\033[33m#%03d/%d of bloc #%d <?= with #%d\033[m", tid, workers, blockIdx.x, dualTid);
    // On évite de taper dans d'autres blocs
    //if (dualTid >= blockDim.x) continue;
    // ALORS on peut réduire:
    {
      //const double tmp = shared_sum_array[dualTid];
      //printf("\n#%03d/%d of bloc #%d <?= with #%d: %0.15e vs %0.15e",tid, workers, blockIdx.x, dualTid,shared_sum_array[tid],shared_sum_array[dualTid]);
      shared_sum_array[tid] += shared_sum_array[dualTid];
    }
    __syncthreads();
  }
  __syncthreads();
  if (tid==0){
    results[bid]=shared_sum_array[0];
    //printf("\n\033[33mSum bloc #%d returned %0.15e\033[m", bid, results[bid]);
    __syncthreads();
  }
}

// *****************************************************************************
__global__
void cudaReduction_sum(const int N,
                       double *reduce_results,
                       const double *data1,
                       const double *data2){ 
  const register int tid = blockDim.x*blockIdx.x + threadIdx.x;
  if (tid>=N) return;
  reduce_sum_kernel(N,reduce_results,data1[tid],data2[tid]);
}

// *****************************************************************************
void reduceSumN(int size,
                const double *d_i1data, const double *d_i2data,
                double *d_odata){
  //printf("\n\033[33m[reduceSumN] d_i1data:\033[m");

  static double *t_i1data=NULL;
  if (!t_i1data){
    cudaMallocManaged(&t_i1data, size*sizeof(double), cudaMemAttachGlobal);
  }
  static double *t_i2data=NULL;
  if (!t_i2data){
    cudaMallocManaged(&t_i2data, size*sizeof(double), cudaMemAttachGlobal);
  }

  //for(int i=0;i<size;i+=1) {
  //printf("\n\033[33m[reduceSumN] d_i1data:\033[m");
  for(int i=size-1;i>=0;i-=1) {
    t_i1data[i]=0.01;
    //printf(" [%f]",t_i1data[i]);//(i+18)*0.123456789);
    //printf(" [%f]",d_i1data[i]);
  }
  //printf("\n\033[33m[reduceSumN] d_i2data:\033[m");
  for(int i=size-1;i>=0;i-=1) {
    //printf(" [%f]",t_idata[i]=(i+18)*0.123456789);
    t_i2data[i]=0.01;
    //printf(" [%f]",t_i2data[i]);//(i+18)*0.123456789);
    //printf(" [%f]",d_i2data[i]);
  }

  const dim3 dimBlock=dim3(BLOCKSIZE,1,1);
  const dim3 dimGrid=dim3(PAD_DIV(size,dimBlock.x),1,1);
  const int reduced_size=(size%CUDA_NB_THREADS_PER_BLOCK)==0?
    (size/CUDA_NB_THREADS_PER_BLOCK):(1+size/CUDA_NB_THREADS_PER_BLOCK);

  //printf("\033[33m[reduceSum] size=%d, reduced_size=%d\n\033[m",size,reduced_size);
  
  static double *results=NULL;
  if (!results){
    cudaMallocManaged(&results, reduced_size*sizeof(double), cudaMemAttachGlobal);
  }
   
  cudaReduction_sum<<<dimGrid,dimBlock>>>(size,
                                          results,
                                          t_i1data,
                                          t_i2data);
  cudaDeviceSynchronize();
  d_odata[0]=0.0;
  for(int i=0;i<reduced_size;i+=1){
    //printf("\n\t\033[33m[reduceSum] reduced=%0.15e, reduce_results[%d/%d]=%0.15e\033[m",d_odata[0],i,results[i],reduced_size);
    d_odata[0]+=results[i];
  }
  //printf("\033[33m[reduceSum] d_odata[0]=%f\n\033[m",d_odata[0]);
}

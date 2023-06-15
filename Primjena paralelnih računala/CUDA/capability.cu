#include <stdio.h>

int main()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);  // daje broj CUDA sposobnih grafičkih kartica
    printf("device count = %d\n", device_count);

    cudaDeviceProp device_prop;
    for(int i = 0; i < device_count; i++) {
        cudaGetDeviceProperties(&device_prop, i); //decide if device has sufficient resources and capabilities
    }

    printf("name =  %s\n", device_prop.name);
    printf("Maksimalan br niti u bloku     = %d\n", device_prop. maxThreadsPerBlock);
    printf("Maksimalan br niti u multiproc = %d\n", device_prop. maxThreadsPerMultiProcessor);
    printf("No streaming multiprocessors = %d\n", device_prop.multiProcessorCount);
    printf("Clock rate = %d (kHz)\n", device_prop.clockRate);
    printf("Max broj niti u bloku x-smjer = %d\n", device_prop.maxThreadsDim[0]);
    printf("Max broj niti u bloku y-smjer = %d\n", device_prop.maxThreadsDim[1]);
    printf("Max broj niti u bloku z-smjer = %d\n", device_prop.maxThreadsDim[2]);
    printf("Max broj blokova po multiproc = %d\n", device_prop.maxBlocksPerMultiProcessor);
    printf("Max broj blokova x-smjer = %d\n", device_prop.maxGridSize[0]);
    printf("Max broj blokova y-smjer = %d\n", device_prop.maxGridSize[1]);
    printf("Max broj blokova z-smjer = %d\n", device_prop.maxGridSize[2]);
    printf("Warp size                = %d\n", device_prop.warpSize);
    printf("Registers per block      = %d (32 bit regs)\n", device_prop.regsPerBlock);
    printf("Registers per multiproc  = %d (32 bit regs)\n", device_prop.regsPerMultiprocessor);
    printf("Količina dijeljene mem po bloku     = %ld (bytes)\n", device_prop.sharedMemPerBlock);
    printf("Količina dijeljene mem po multiproc = %ld (bytes)\n", device_prop.sharedMemPerMultiprocessor);
//    printf("= %d\n", device_prop.);

    return 0;
}

/*
  struct cudaDeviceProp {
              char name[256];
              cudaUUID_t uuid;
              size_t totalGlobalMem;
              size_t sharedMemPerBlock;
              int regsPerBlock;
              int warpSize;
              size_t memPitch;
              int maxThreadsPerBlock;
              int maxThreadsDim[3];
              int maxGridSize[3];
              int clockRate;
              size_t totalConstMem;
              int major;
              int minor;
              size_t textureAlignment;
              size_t texturePitchAlignment;
              int deviceOverlap;
              int multiProcessorCount;
              int kernelExecTimeoutEnabled;
              int integrated;
              int canMapHostMemory;
              int computeMode;
              int maxTexture1D;
              int maxTexture1DMipmap;
              int maxTexture1DLinear;
              int maxTexture2D[2];
              int maxTexture2DMipmap[2];
              int maxTexture2DLinear[3];
              int maxTexture2DGather[2];
              int maxTexture3D[3];
              int maxTexture3DAlt[3];
              int maxTextureCubemap;
              int maxTexture1DLayered[2];
              int maxTexture2DLayered[3];
              int maxTextureCubemapLayered[2];
              int maxSurface1D;
              int maxSurface2D[2];
              int maxSurface3D[3];
              int maxSurface1DLayered[2];
              int maxSurface2DLayered[3];
              int maxSurfaceCubemap;
              int maxSurfaceCubemapLayered[2];
              size_t surfaceAlignment;
              int concurrentKernels;
              int ECCEnabled;
              int pciBusID;
              int pciDeviceID;
              int pciDomainID;
              int tccDriver;
              int asyncEngineCount;
              int unifiedAddressing;
              int memoryClockRate;
              int memoryBusWidth;
              int l2CacheSize;
              int persistingL2CacheMaxSize;
              int maxThreadsPerMultiProcessor;
              int streamPrioritiesSupported;
              int globalL1CacheSupported;
              int localL1CacheSupported;
              size_t sharedMemPerMultiprocessor;
              int regsPerMultiprocessor;
              int managedMemory;
              int isMultiGpuBoard;
              int multiGpuBoardGroupID;
              int singleToDoublePrecisionPerfRatio;
              int pageableMemoryAccess;
              int concurrentManagedAccess;
              int computePreemptionSupported;
              int canUseHostPointerForRegisteredMem;
              int cooperativeLaunch;
              int cooperativeMultiDeviceLaunch;
              int pageableMemoryAccessUsesHostPageTables;
              int directManagedMemAccessFromHost;
              int accessPolicyMaxWindowSize;
          }
*/

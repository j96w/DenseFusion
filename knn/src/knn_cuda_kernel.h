#ifndef _MATHUTIL_CUDA_KERNEL
#define _MATHUTIL_CUDA_KERNEL

#define IDX2D(i, j, dj) (dj * i + j)
#define IDX3D(i, j, k, dj, dk) (IDX2D(IDX2D(i, j, dj), k, dk))

#define BLOCK 512
#define MAX_STREAMS 512

#ifdef __cplusplus
extern "C" {
#endif

void knn_device(float* ref_dev, int ref_width,
    float* query_dev, int query_width,
    int height, int k, float* dist_dev, long* ind_dev, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

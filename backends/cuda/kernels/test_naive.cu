// test kernel with known memory issues
__global__ void naive_matrix_op(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        C[row * n + col] = A[col * n + row] + B[row * n + col];
    }
}

extern "C" __global__
void vectorAdd(const float *a, const float *b, float *c) {
      // Vector index
      int nIndex = threadIdx.x;
      c[nIndex] = a[nIndex] + b[nIndex];
}

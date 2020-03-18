__kernel
void dotProduct(global float* restrict output, int n4, const global float* restrict vecA5, const global float* restrict vecB6, global float* restrict x36, global float* restrict x38){
  /* Start of moved local vars */
  /* End of moved local vars */
  /* mapSeq */
  for (int i_51 = 0;(i_51 < n4);i_51 = (1 + i_51)) {
    x38[i_51] = (vecA5[i_51] * vecB6[i_51]);
  }

  /* oclReduceSeq */
  x36[0] = 0.0f;
  for (int i_52 = 0;(i_52 < n4);i_52 = (1 + i_52)) {
    x36[0] = (x36[0] + x38[i_52]);
  }

  output[0] = x36[0];
}

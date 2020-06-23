#include <stdint.h>

#include <stdint.h>


void gemm(float* output, int n4, int m5, int k6, float* matA7, float* matB8, float* matC9, float alpha10, float beta11){
  /* mapSeq */
  for (int i_56 = 0;(i_56 < n4);i_56 = (1 + i_56)) {
    /* mapSeq */
    for (int i_57 = 0;(i_57 < k6);i_57 = (1 + i_57)) {
      {
        float x35[m5];
        /* mapSeq */
        for (int i_58 = 0;(i_58 < m5);i_58 = (1 + i_58)) {
          x35[i_58] = (matA7[(i_58 + (i_56 * m5))] * matB8[(i_57 + (i_58 * k6))]);
        }
        
        /* reduceSeq */
        {
          float x33;
          x33 = 0.0f;
          for (int i_59 = 0;(i_59 < m5);i_59 = (1 + i_59)) {
            x33 = (x33 + x35[i_59]);
          }
          
          output[(i_57 + (i_56 * k6))] = ((alpha10 * x33) + (beta11 * matC9[(i_57 + (i_56 * k6))]));
        }
        
      }
      
    }
    
  }
  
}


void executegemm (void** parameter) {
    gemm(
        parameter[0],
        *((int32_t*) parameter[1]),
        *((int32_t*) parameter[2]),
        *((int32_t*) parameter[3]),
        parameter[4],
        parameter[5],
        parameter[6],
        *((float*) parameter[7]),
        *((float*) parameter[8]));
}

struct opfngemm{ void (*op)(void** parameter);};

struct opfngemm opgemm = {.op = executegemm};

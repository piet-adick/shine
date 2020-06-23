#include <stdint.h>

#include <stdint.h>


void dotProduct(float* output, int n4, float* vecA5, float* vecB6){
  {
    float x16[n4];
    /* mapSeq */
    for (int i_27 = 0;(i_27 < n4);i_27 = (1 + i_27)) {
      x16[i_27] = (vecA5[i_27] * vecB6[i_27]);
    }
    
    /* reduceSeq */
    {
      float x14;
      x14 = 0.0f;
      for (int i_28 = 0;(i_28 < n4);i_28 = (1 + i_28)) {
        x14 = (x14 + x16[i_28]);
      }
      
      output[0] = x14;
    }
    
  }
  
}


void executedotProduct (void** parameter) {
    dotProduct(
        parameter[0],
        *((int32_t*) parameter[1]),
        parameter[2],
        parameter[3]);
}

struct opfndotProduct{ void (*op)(void** parameter);};

struct opfndotProduct opdotProduct = {.op = executedotProduct};

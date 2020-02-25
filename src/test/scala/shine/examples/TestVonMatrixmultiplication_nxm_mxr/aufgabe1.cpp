#include <iostream>
using namespace std;


void matrixMult(float* output, int r6, int m5, int n4, float* MatrixA9, float* MatrixB10){
  /* mapSeq */
  for (int i_41 = 0;(i_41 < n4);i_41 = (1 + i_41)) {
    /* mapSeq */
    for (int i_42 = 0;(i_42 < r6);i_42 = (1 + i_42)) {
      {
        float x27[m5];
        /* mapSeq */
        for (int i_43 = 0;(i_43 < m5);i_43 = (1 + i_43)) {
          x27[i_43] = (MatrixB10[(i_42 + (i_43 * r6))] * MatrixA9[(i_43 + (i_41 * m5))]);
        }
        
        {
          float x25;
          x25 = 0.0f;
          /* reduceSeq */
          for (int i_44 = 0;(i_44 < m5);i_44 = (1 + i_44)) {
            x25 = (x25 + x27[i_44]);
          }
          
          output[(i_42 + (i_41 * r6))] = x25;
        }
        
      }
      
    }
    
  }
  
}


void printMatrix(int n, int m, float* matrix){
    for(int i = 0; i<n; ++i){
        printf("\n( ");
        for(int j = 0; j<m; ++j){
            printf("%f, " ,matrix[i*m + j]);
        }
        printf(" )");
    }
    printf("\n");
}


void einfach(){
      cout << "Hallo Welt!" << endl;
    float matrixA[9] = {
        1.0f , 2.0f , 3.0f,
        4.0f , 7.0f, 11.0f,
        3.0f , 5.0f , 9.0f
    };
    float matrixB[9] = {
        -8.0f , 3.0f , -1.0f, 
         3.0f , 0.0f , -1.0f,
         1.0f , -1.0f ,  1.0f
    };
    float* erg =(float*) malloc(sizeof(float) * 3 *3);
    matrixMult(erg, 3, 3, 3, matrixA, matrixB);
    printMatrix(3, 3, matrixA);
    printMatrix(3, 3, matrixB);
    printMatrix(3, 3, erg);


}

int main(void){

    cout << "Hallo Welt!" << endl;
    float matrixA[4*3] = {
      25.7f , 4.3f, 2.9f, 1.2f,
      4.0f  , 98.0f, 99.0f, 1.3f,
      -98.0f, -14.0f, 25.0f, -11.0f
    };
    float matrixB[2*4] = {
        -1.0f, 1.0f,
        12.8f, 2.2f,
        11.7f, 5.5f,
        11.2f, 9.8f
    };
    float* erg =(float*) malloc(sizeof(float) * 3 *3);
    matrixMult(erg, 2, 4, 3, matrixA, matrixB);
    printMatrix(3, 4, matrixA);
    printMatrix(4, 2, matrixB);
    printMatrix(3, 2, erg);

    //Ergebnis muss:
    /*
    ({{76.71, 62.87},
     {2423.26, 776.84},
      {88.10, -99.10}})
    */
}


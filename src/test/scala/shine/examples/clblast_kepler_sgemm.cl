#ifndef Tuple_float_float_DEFINED
#define Tuple_float_float_DEFINED
typedef struct {
  float _0;
  float _1;
} Tuple_float_float;
#endif

#ifndef Tuple_float2_float2_DEFINED
#define Tuple_float2_float2_DEFINED
typedef struct {
  float2 _0;
  float2 _1;
} Tuple_float2_float2;
#endif

float2 idfloat2(float2 x){
  { return x; }
}
float2 add2(float2 x, float2 y){
  { return x+y; }
}
float2 mult2(float2 l, float2 r){
  { return l * r; }
}
float idfloat(float x){
  { return x; }
}
float add(float x, float y){
  { return x+y; }
}
float mult(float l, float r){
  { return l * r; }
}
kernel void clblast_kepler_sgemm(const global float* restrict v__141, const global float* restrict v__142, const global float* restrict v__143, float v__144, float v__145, global float* v__180, int v_K_2, int v_M_1, int v_N_0){ 
 /* Static local memory */
 local float v__154[2048];
 local float v__153[1024];
 
#ifndef WORKGROUP_GUARD
#define WORKGROUP_GUARD
#endif
WORKGROUP_GUARD
{
  /* Typed Value memory */
  float v__149;
  /* Private Memory */
  float v__150_0;
  float v__150_1;
  float v__150_2;
  float v__150_3;
  float v__150_4;
  float v__150_5;
  float v__150_6;
  float v__150_7;
  float v__150_8;
  float v__150_9;
  float v__150_10;
  float v__150_11;
  float v__150_12;
  float v__150_13;
  float v__150_14;
  float v__150_15;
  float v__150_16;
  float v__150_17;
  float v__150_18;
  float v__150_19;
  float v__150_20;
  float v__150_21;
  float v__150_22;
  float v__150_23;
  float v__150_24;
  float v__150_25;
  float v__150_26;
  float v__150_27;
  float v__150_28;
  float v__150_29;
  float v__150_30;
  float v__150_31;
  float v__150_32;
  float v__150_33;
  float v__150_34;
  float v__150_35;
  float v__150_36;
  float v__150_37;
  float v__150_38;
  float v__150_39;
  float v__150_40;
  float v__150_41;
  float v__150_42;
  float v__150_43;
  float v__150_44;
  float v__150_45;
  float v__150_46;
  float v__150_47;
  float v__150_48;
  float v__150_49;
  float v__150_50;
  float v__150_51;
  float v__150_52;
  float v__150_53;
  float v__150_54;
  float v__150_55;
  float v__150_56;
  float v__150_57;
  float v__150_58;
  float v__150_59;
  float v__150_60;
  float v__150_61;
  float v__150_62;
  float v__150_63;
  
  float v__160_0;
  float v__160_1;
  float v__160_2;
  float v__160_3;
  float v__160_4;
  float v__160_5;
  float v__160_6;
  float v__160_7;
  
  float v__161_0;
  float v__161_1;
  float v__161_2;
  float v__161_3;
  float v__161_4;
  float v__161_5;
  float v__161_6;
  float v__161_7;
  
  float v__166_0;
  float v__166_1;
  float v__166_2;
  float v__166_3;
  float v__166_4;
  float v__166_5;
  float v__166_6;
  float v__166_7;
  float v__166_8;
  float v__166_9;
  float v__166_10;
  float v__166_11;
  float v__166_12;
  float v__166_13;
  float v__166_14;
  float v__166_15;
  float v__166_16;
  float v__166_17;
  float v__166_18;
  float v__166_19;
  float v__166_20;
  float v__166_21;
  float v__166_22;
  float v__166_23;
  float v__166_24;
  float v__166_25;
  float v__166_26;
  float v__166_27;
  float v__166_28;
  float v__166_29;
  float v__166_30;
  float v__166_31;
  float v__166_32;
  float v__166_33;
  float v__166_34;
  float v__166_35;
  float v__166_36;
  float v__166_37;
  float v__166_38;
  float v__166_39;
  float v__166_40;
  float v__166_41;
  float v__166_42;
  float v__166_43;
  float v__166_44;
  float v__166_45;
  float v__166_46;
  float v__166_47;
  float v__166_48;
  float v__166_49;
  float v__166_50;
  float v__166_51;
  float v__166_52;
  float v__166_53;
  float v__166_54;
  float v__166_55;
  float v__166_56;
  float v__166_57;
  float v__166_58;
  float v__166_59;
  float v__166_60;
  float v__166_61;
  float v__166_62;
  float v__166_63;
  
  float2 v__175_0;
  float2 v__175_1;
  float2 v__175_2;
  float2 v__175_3;
  float2 v__175_4;
  float2 v__175_5;
  float2 v__175_6;
  float2 v__175_7;
  float2 v__175_8;
  float2 v__175_9;
  float2 v__175_10;
  float2 v__175_11;
  float2 v__175_12;
  float2 v__175_13;
  float2 v__175_14;
  float2 v__175_15;
  float2 v__175_16;
  float2 v__175_17;
  float2 v__175_18;
  float2 v__175_19;
  float2 v__175_20;
  float2 v__175_21;
  float2 v__175_22;
  float2 v__175_23;
  float2 v__175_24;
  float2 v__175_25;
  float2 v__175_26;
  float2 v__175_27;
  float2 v__175_28;
  float2 v__175_29;
  float2 v__175_30;
  float2 v__175_31;
  
  float2 v__178_0;
  float2 v__178_1;
  float2 v__178_2;
  float2 v__178_3;
  float2 v__178_4;
  float2 v__178_5;
  float2 v__178_6;
  float2 v__178_7;
  float2 v__178_8;
  float2 v__178_9;
  float2 v__178_10;
  float2 v__178_11;
  float2 v__178_12;
  float2 v__178_13;
  float2 v__178_14;
  float2 v__178_15;
  float2 v__178_16;
  float2 v__178_17;
  float2 v__178_18;
  float2 v__178_19;
  float2 v__178_20;
  float2 v__178_21;
  float2 v__178_22;
  float2 v__178_23;
  float2 v__178_24;
  float2 v__178_25;
  float2 v__178_26;
  float2 v__178_27;
  float2 v__178_28;
  float2 v__178_29;
  float2 v__178_30;
  float2 v__178_31;
  
  /* iteration count is exactly 1, no loop emitted */
  {
    int v_wg_id_111 = get_group_id(1);
    /* iteration count is exactly 1, no loop emitted */
    {
      int v_wg_id_112 = get_group_id(0);
      float v_tmp_416 = 0.0f;
      v__149 = v_tmp_416;
      /* unroll */
      /* unroll */
      /* map_seq */
      /* unroll */
      /* map_seq */
      /* unroll */
      v__150_0 = idfloat(v__149);
      v__150_1 = idfloat(v__149);
      v__150_2 = idfloat(v__149);
      v__150_3 = idfloat(v__149);
      v__150_4 = idfloat(v__149);
      v__150_5 = idfloat(v__149);
      v__150_6 = idfloat(v__149);
      v__150_7 = idfloat(v__149);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__150_8 = idfloat(v__149);
      v__150_9 = idfloat(v__149);
      v__150_10 = idfloat(v__149);
      v__150_11 = idfloat(v__149);
      v__150_12 = idfloat(v__149);
      v__150_13 = idfloat(v__149);
      v__150_14 = idfloat(v__149);
      v__150_15 = idfloat(v__149);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__150_16 = idfloat(v__149);
      v__150_17 = idfloat(v__149);
      v__150_18 = idfloat(v__149);
      v__150_19 = idfloat(v__149);
      v__150_20 = idfloat(v__149);
      v__150_21 = idfloat(v__149);
      v__150_22 = idfloat(v__149);
      v__150_23 = idfloat(v__149);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__150_24 = idfloat(v__149);
      v__150_25 = idfloat(v__149);
      v__150_26 = idfloat(v__149);
      v__150_27 = idfloat(v__149);
      v__150_28 = idfloat(v__149);
      v__150_29 = idfloat(v__149);
      v__150_30 = idfloat(v__149);
      v__150_31 = idfloat(v__149);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__150_32 = idfloat(v__149);
      v__150_33 = idfloat(v__149);
      v__150_34 = idfloat(v__149);
      v__150_35 = idfloat(v__149);
      v__150_36 = idfloat(v__149);
      v__150_37 = idfloat(v__149);
      v__150_38 = idfloat(v__149);
      v__150_39 = idfloat(v__149);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__150_40 = idfloat(v__149);
      v__150_41 = idfloat(v__149);
      v__150_42 = idfloat(v__149);
      v__150_43 = idfloat(v__149);
      v__150_44 = idfloat(v__149);
      v__150_45 = idfloat(v__149);
      v__150_46 = idfloat(v__149);
      v__150_47 = idfloat(v__149);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__150_48 = idfloat(v__149);
      v__150_49 = idfloat(v__149);
      v__150_50 = idfloat(v__149);
      v__150_51 = idfloat(v__149);
      v__150_52 = idfloat(v__149);
      v__150_53 = idfloat(v__149);
      v__150_54 = idfloat(v__149);
      v__150_55 = idfloat(v__149);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__150_56 = idfloat(v__149);
      v__150_57 = idfloat(v__149);
      v__150_58 = idfloat(v__149);
      v__150_59 = idfloat(v__149);
      v__150_60 = idfloat(v__149);
      v__150_61 = idfloat(v__149);
      v__150_62 = idfloat(v__149);
      v__150_63 = idfloat(v__149);
      /* end unroll */
      /* end map_seq */
      /* end unroll */
      /* end map_seq */
      /* end unroll */
      /* end unroll */
      /* reduce_seq */
      for (int v_i_117 = 0;v_i_117<(v_K_2 / (16));v_i_117 = (1 + v_i_117)){
        /* iteration count is exactly 1, no loop emitted */
        {
          int v_l_id_122 = get_local_id(1);
          for (int v_l_id_123 = get_local_id(0);v_l_id_123<64;v_l_id_123 = (16 + v_l_id_123)){
            vstore2(idfloat2(vload2((((((2 * v_l_id_123) % 64) + (v_M_1 * (v_l_id_123 / 32))) / 2) + (8 * v_M_1 * v_i_117) + (v_M_1 * v_l_id_122) + (32 * ((v_wg_id_111 + (v_M_1 * v_l_id_122 / (32))) % (v_M_1 / (64))))),v__141)),(v_l_id_123 + (64 * v_l_id_122)),v__153);;
          }
          barrier(CLK_LOCAL_MEM_FENCE);
          
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int v_l_id_124 = get_local_id(1);v_l_id_124<16;v_l_id_124 = (8 + v_l_id_124)){
          for (int v_l_id_125 = get_local_id(0);v_l_id_125<64;v_l_id_125 = (16 + v_l_id_125)){
            vstore2(idfloat2(vload2((v_l_id_125 + ((v_N_0 * v_l_id_124) / 2) + (8 * v_N_0 * v_i_117) + (64 * v_wg_id_112)),v__142)),(v_l_id_125 + (64 * v_l_id_124)),v__154);;
          }
          barrier(CLK_LOCAL_MEM_FENCE);
          
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        /* unroll */
        /* unroll */
        /* reduce_seq */
        for (int v_i_128 = 0;v_i_128<16;v_i_128 = (1 + v_i_128)){
          /* map_seq */
          /* unroll */
          v__160_0 = idfloat(v__153[((8 * get_local_id(1)) + (64 * v_i_128))]);
          v__160_1 = idfloat(v__153[(1 + (8 * get_local_id(1)) + (64 * v_i_128))]);
          v__160_2 = idfloat(v__153[(2 + (8 * get_local_id(1)) + (64 * v_i_128))]);
          v__160_3 = idfloat(v__153[(3 + (8 * get_local_id(1)) + (64 * v_i_128))]);
          v__160_4 = idfloat(v__153[(4 + (8 * get_local_id(1)) + (64 * v_i_128))]);
          v__160_5 = idfloat(v__153[(5 + (8 * get_local_id(1)) + (64 * v_i_128))]);
          v__160_6 = idfloat(v__153[(6 + (8 * get_local_id(1)) + (64 * v_i_128))]);
          v__160_7 = idfloat(v__153[(7 + (8 * get_local_id(1)) + (64 * v_i_128))]);
          /* end unroll */
          /* end map_seq */
          /* map_seq */
          /* unroll */
          v__161_0 = idfloat(v__154[((2 * get_local_id(0)) + (128 * v_i_128))]);
          v__161_1 = idfloat(v__154[(1 + (2 * get_local_id(0)) + (128 * v_i_128))]);
          v__161_2 = idfloat(v__154[(32 + (2 * get_local_id(0)) + (128 * v_i_128))]);
          v__161_3 = idfloat(v__154[(33 + (2 * get_local_id(0)) + (128 * v_i_128))]);
          v__161_4 = idfloat(v__154[(64 + (2 * get_local_id(0)) + (128 * v_i_128))]);
          v__161_5 = idfloat(v__154[(65 + (2 * get_local_id(0)) + (128 * v_i_128))]);
          v__161_6 = idfloat(v__154[(96 + (2 * get_local_id(0)) + (128 * v_i_128))]);
          v__161_7 = idfloat(v__154[(97 + (2 * get_local_id(0)) + (128 * v_i_128))]);
          /* end unroll */
          /* end map_seq */
          /* map_seq */
          /* unroll */
          /* map_seq */
          /* unroll */
          v__166_0 = mult(v__160_0, v__161_0);
          v__150_0 = add(v__150_0, v__166_0);
          v__166_1 = mult(v__160_0, v__161_1);
          v__150_1 = add(v__150_1, v__166_1);
          v__166_2 = mult(v__160_0, v__161_2);
          v__150_2 = add(v__150_2, v__166_2);
          v__166_3 = mult(v__160_0, v__161_3);
          v__150_3 = add(v__150_3, v__166_3);
          v__166_4 = mult(v__160_0, v__161_4);
          v__150_4 = add(v__150_4, v__166_4);
          v__166_5 = mult(v__160_0, v__161_5);
          v__150_5 = add(v__150_5, v__166_5);
          v__166_6 = mult(v__160_0, v__161_6);
          v__150_6 = add(v__150_6, v__166_6);
          v__166_7 = mult(v__160_0, v__161_7);
          v__150_7 = add(v__150_7, v__166_7);
          /* end unroll */
          /* end map_seq */
          /* map_seq */
          /* unroll */
          v__166_8 = mult(v__160_1, v__161_0);
          v__150_8 = add(v__150_8, v__166_8);
          v__166_9 = mult(v__160_1, v__161_1);
          v__150_9 = add(v__150_9, v__166_9);
          v__166_10 = mult(v__160_1, v__161_2);
          v__150_10 = add(v__150_10, v__166_10);
          v__166_11 = mult(v__160_1, v__161_3);
          v__150_11 = add(v__150_11, v__166_11);
          v__166_12 = mult(v__160_1, v__161_4);
          v__150_12 = add(v__150_12, v__166_12);
          v__166_13 = mult(v__160_1, v__161_5);
          v__150_13 = add(v__150_13, v__166_13);
          v__166_14 = mult(v__160_1, v__161_6);
          v__150_14 = add(v__150_14, v__166_14);
          v__166_15 = mult(v__160_1, v__161_7);
          v__150_15 = add(v__150_15, v__166_15);
          /* end unroll */
          /* end map_seq */
          /* map_seq */
          /* unroll */
          v__166_16 = mult(v__160_2, v__161_0);
          v__150_16 = add(v__150_16, v__166_16);
          v__166_17 = mult(v__160_2, v__161_1);
          v__150_17 = add(v__150_17, v__166_17);
          v__166_18 = mult(v__160_2, v__161_2);
          v__150_18 = add(v__150_18, v__166_18);
          v__166_19 = mult(v__160_2, v__161_3);
          v__150_19 = add(v__150_19, v__166_19);
          v__166_20 = mult(v__160_2, v__161_4);
          v__150_20 = add(v__150_20, v__166_20);
          v__166_21 = mult(v__160_2, v__161_5);
          v__150_21 = add(v__150_21, v__166_21);
          v__166_22 = mult(v__160_2, v__161_6);
          v__150_22 = add(v__150_22, v__166_22);
          v__166_23 = mult(v__160_2, v__161_7);
          v__150_23 = add(v__150_23, v__166_23);
          /* end unroll */
          /* end map_seq */
          /* map_seq */
          /* unroll */
          v__166_24 = mult(v__160_3, v__161_0);
          v__150_24 = add(v__150_24, v__166_24);
          v__166_25 = mult(v__160_3, v__161_1);
          v__150_25 = add(v__150_25, v__166_25);
          v__166_26 = mult(v__160_3, v__161_2);
          v__150_26 = add(v__150_26, v__166_26);
          v__166_27 = mult(v__160_3, v__161_3);
          v__150_27 = add(v__150_27, v__166_27);
          v__166_28 = mult(v__160_3, v__161_4);
          v__150_28 = add(v__150_28, v__166_28);
          v__166_29 = mult(v__160_3, v__161_5);
          v__150_29 = add(v__150_29, v__166_29);
          v__166_30 = mult(v__160_3, v__161_6);
          v__150_30 = add(v__150_30, v__166_30);
          v__166_31 = mult(v__160_3, v__161_7);
          v__150_31 = add(v__150_31, v__166_31);
          /* end unroll */
          /* end map_seq */
          /* map_seq */
          /* unroll */
          v__166_32 = mult(v__160_4, v__161_0);
          v__150_32 = add(v__150_32, v__166_32);
          v__166_33 = mult(v__160_4, v__161_1);
          v__150_33 = add(v__150_33, v__166_33);
          v__166_34 = mult(v__160_4, v__161_2);
          v__150_34 = add(v__150_34, v__166_34);
          v__166_35 = mult(v__160_4, v__161_3);
          v__150_35 = add(v__150_35, v__166_35);
          v__166_36 = mult(v__160_4, v__161_4);
          v__150_36 = add(v__150_36, v__166_36);
          v__166_37 = mult(v__160_4, v__161_5);
          v__150_37 = add(v__150_37, v__166_37);
          v__166_38 = mult(v__160_4, v__161_6);
          v__150_38 = add(v__150_38, v__166_38);
          v__166_39 = mult(v__160_4, v__161_7);
          v__150_39 = add(v__150_39, v__166_39);
          /* end unroll */
          /* end map_seq */
          /* map_seq */
          /* unroll */
          v__166_40 = mult(v__160_5, v__161_0);
          v__150_40 = add(v__150_40, v__166_40);
          v__166_41 = mult(v__160_5, v__161_1);
          v__150_41 = add(v__150_41, v__166_41);
          v__166_42 = mult(v__160_5, v__161_2);
          v__150_42 = add(v__150_42, v__166_42);
          v__166_43 = mult(v__160_5, v__161_3);
          v__150_43 = add(v__150_43, v__166_43);
          v__166_44 = mult(v__160_5, v__161_4);
          v__150_44 = add(v__150_44, v__166_44);
          v__166_45 = mult(v__160_5, v__161_5);
          v__150_45 = add(v__150_45, v__166_45);
          v__166_46 = mult(v__160_5, v__161_6);
          v__150_46 = add(v__150_46, v__166_46);
          v__166_47 = mult(v__160_5, v__161_7);
          v__150_47 = add(v__150_47, v__166_47);
          /* end unroll */
          /* end map_seq */
          /* map_seq */
          /* unroll */
          v__166_48 = mult(v__160_6, v__161_0);
          v__150_48 = add(v__150_48, v__166_48);
          v__166_49 = mult(v__160_6, v__161_1);
          v__150_49 = add(v__150_49, v__166_49);
          v__166_50 = mult(v__160_6, v__161_2);
          v__150_50 = add(v__150_50, v__166_50);
          v__166_51 = mult(v__160_6, v__161_3);
          v__150_51 = add(v__150_51, v__166_51);
          v__166_52 = mult(v__160_6, v__161_4);
          v__150_52 = add(v__150_52, v__166_52);
          v__166_53 = mult(v__160_6, v__161_5);
          v__150_53 = add(v__150_53, v__166_53);
          v__166_54 = mult(v__160_6, v__161_6);
          v__150_54 = add(v__150_54, v__166_54);
          v__166_55 = mult(v__160_6, v__161_7);
          v__150_55 = add(v__150_55, v__166_55);
          /* end unroll */
          /* end map_seq */
          /* map_seq */
          /* unroll */
          v__166_56 = mult(v__160_7, v__161_0);
          v__150_56 = add(v__150_56, v__166_56);
          v__166_57 = mult(v__160_7, v__161_1);
          v__150_57 = add(v__150_57, v__166_57);
          v__166_58 = mult(v__160_7, v__161_2);
          v__150_58 = add(v__150_58, v__166_58);
          v__166_59 = mult(v__160_7, v__161_3);
          v__150_59 = add(v__150_59, v__166_59);
          v__166_60 = mult(v__160_7, v__161_4);
          v__150_60 = add(v__150_60, v__166_60);
          v__166_61 = mult(v__160_7, v__161_5);
          v__150_61 = add(v__150_61, v__166_61);
          v__166_62 = mult(v__160_7, v__161_6);
          v__150_62 = add(v__150_62, v__166_62);
          v__166_63 = mult(v__160_7, v__161_7);
          v__150_63 = add(v__150_63, v__166_63);
          /* end unroll */
          /* end map_seq */
          /* end unroll */
          /* end map_seq */
        }
        /* end reduce_seq */
        /* map_seq */
        /* unroll */
        /* end unroll */
        /* end map_seq */
        /* end unroll */
        /* end unroll */
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        
      }
      /* end reduce_seq */
      /* map_seq */
      /* unroll */
      /* unroll */
      /* unroll */
      /* map_seq */
      /* unroll */
      /* map_seq */
      /* unroll */
      v__175_0 = mult2((float2)(v__150_0, v__150_1), (float)v__144);
      v__178_0 = mult2(vload2(((4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_0, v__178_0),((4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_1 = mult2((float2)(v__150_2, v__150_3), (float)v__144);
      v__178_1 = mult2(vload2((16 + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_1, v__178_1),(16 + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_2 = mult2((float2)(v__150_4, v__150_5), (float)v__144);
      v__178_2 = mult2(vload2((32 + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_2, v__178_2),(32 + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_3 = mult2((float2)(v__150_6, v__150_7), (float)v__144);
      v__178_3 = mult2(vload2((48 + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_3, v__178_3),(48 + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__175_4 = mult2((float2)(v__150_8, v__150_9), (float)v__144);
      v__178_4 = mult2(vload2(((v_N_0 / 2) + (32 * v_N_0 * v_wg_id_111) + (4 * v_N_0 * get_local_id(1)) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_4, v__178_4),((v_N_0 / 2) + (32 * v_N_0 * v_wg_id_111) + (4 * v_N_0 * get_local_id(1)) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_5 = mult2((float2)(v__150_10, v__150_11), (float)v__144);
      v__178_5 = mult2(vload2((16 + (v_N_0 / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_5, v__178_5),(16 + (v_N_0 / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_6 = mult2((float2)(v__150_12, v__150_13), (float)v__144);
      v__178_6 = mult2(vload2((32 + (v_N_0 / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_6, v__178_6),(32 + (v_N_0 / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_7 = mult2((float2)(v__150_14, v__150_15), (float)v__144);
      v__178_7 = mult2(vload2((48 + (v_N_0 / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_7, v__178_7),(48 + (v_N_0 / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__175_8 = mult2((float2)(v__150_16, v__150_17), (float)v__144);
      v__178_8 = mult2(vload2((v_N_0 + (32 * v_N_0 * v_wg_id_111) + (4 * v_N_0 * get_local_id(1)) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_8, v__178_8),(v_N_0 + (32 * v_N_0 * v_wg_id_111) + (4 * v_N_0 * get_local_id(1)) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_9 = mult2((float2)(v__150_18, v__150_19), (float)v__144);
      v__178_9 = mult2(vload2((16 + v_N_0 + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_9, v__178_9),(16 + v_N_0 + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_10 = mult2((float2)(v__150_20, v__150_21), (float)v__144);
      v__178_10 = mult2(vload2((32 + v_N_0 + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_10, v__178_10),(32 + v_N_0 + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_11 = mult2((float2)(v__150_22, v__150_23), (float)v__144);
      v__178_11 = mult2(vload2((48 + v_N_0 + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_11, v__178_11),(48 + v_N_0 + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__175_12 = mult2((float2)(v__150_24, v__150_25), (float)v__144);
      v__178_12 = mult2(vload2((((3 * v_N_0) / 2) + (32 * v_N_0 * v_wg_id_111) + (4 * v_N_0 * get_local_id(1)) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_12, v__178_12),(((3 * v_N_0) / 2) + (32 * v_N_0 * v_wg_id_111) + (4 * v_N_0 * get_local_id(1)) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_13 = mult2((float2)(v__150_26, v__150_27), (float)v__144);
      v__178_13 = mult2(vload2((16 + ((3 * v_N_0) / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_13, v__178_13),(16 + ((3 * v_N_0) / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_14 = mult2((float2)(v__150_28, v__150_29), (float)v__144);
      v__178_14 = mult2(vload2((32 + ((3 * v_N_0) / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_14, v__178_14),(32 + ((3 * v_N_0) / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_15 = mult2((float2)(v__150_30, v__150_31), (float)v__144);
      v__178_15 = mult2(vload2((48 + ((3 * v_N_0) / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_15, v__178_15),(48 + ((3 * v_N_0) / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__175_16 = mult2((float2)(v__150_32, v__150_33), (float)v__144);
      v__178_16 = mult2(vload2(((2 * v_N_0) + (32 * v_N_0 * v_wg_id_111) + (4 * v_N_0 * get_local_id(1)) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_16, v__178_16),((2 * v_N_0) + (32 * v_N_0 * v_wg_id_111) + (4 * v_N_0 * get_local_id(1)) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_17 = mult2((float2)(v__150_34, v__150_35), (float)v__144);
      v__178_17 = mult2(vload2((16 + (2 * v_N_0) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_17, v__178_17),(16 + (2 * v_N_0) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_18 = mult2((float2)(v__150_36, v__150_37), (float)v__144);
      v__178_18 = mult2(vload2((32 + (2 * v_N_0) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_18, v__178_18),(32 + (2 * v_N_0) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_19 = mult2((float2)(v__150_38, v__150_39), (float)v__144);
      v__178_19 = mult2(vload2((48 + (2 * v_N_0) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_19, v__178_19),(48 + (2 * v_N_0) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__175_20 = mult2((float2)(v__150_40, v__150_41), (float)v__144);
      v__178_20 = mult2(vload2((((5 * v_N_0) / 2) + (32 * v_N_0 * v_wg_id_111) + (4 * v_N_0 * get_local_id(1)) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_20, v__178_20),(((5 * v_N_0) / 2) + (32 * v_N_0 * v_wg_id_111) + (4 * v_N_0 * get_local_id(1)) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_21 = mult2((float2)(v__150_42, v__150_43), (float)v__144);
      v__178_21 = mult2(vload2((16 + ((5 * v_N_0) / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_21, v__178_21),(16 + ((5 * v_N_0) / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_22 = mult2((float2)(v__150_44, v__150_45), (float)v__144);
      v__178_22 = mult2(vload2((32 + ((5 * v_N_0) / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_22, v__178_22),(32 + ((5 * v_N_0) / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_23 = mult2((float2)(v__150_46, v__150_47), (float)v__144);
      v__178_23 = mult2(vload2((48 + ((5 * v_N_0) / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_23, v__178_23),(48 + ((5 * v_N_0) / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__175_24 = mult2((float2)(v__150_48, v__150_49), (float)v__144);
      v__178_24 = mult2(vload2(((3 * v_N_0) + (32 * v_N_0 * v_wg_id_111) + (4 * v_N_0 * get_local_id(1)) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_24, v__178_24),((3 * v_N_0) + (32 * v_N_0 * v_wg_id_111) + (4 * v_N_0 * get_local_id(1)) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_25 = mult2((float2)(v__150_50, v__150_51), (float)v__144);
      v__178_25 = mult2(vload2((16 + (3 * v_N_0) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_25, v__178_25),(16 + (3 * v_N_0) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_26 = mult2((float2)(v__150_52, v__150_53), (float)v__144);
      v__178_26 = mult2(vload2((32 + (3 * v_N_0) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_26, v__178_26),(32 + (3 * v_N_0) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_27 = mult2((float2)(v__150_54, v__150_55), (float)v__144);
      v__178_27 = mult2(vload2((48 + (3 * v_N_0) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_27, v__178_27),(48 + (3 * v_N_0) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__175_28 = mult2((float2)(v__150_56, v__150_57), (float)v__144);
      v__178_28 = mult2(vload2((((7 * v_N_0) / 2) + (32 * v_N_0 * v_wg_id_111) + (4 * v_N_0 * get_local_id(1)) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_28, v__178_28),(((7 * v_N_0) / 2) + (32 * v_N_0 * v_wg_id_111) + (4 * v_N_0 * get_local_id(1)) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_29 = mult2((float2)(v__150_58, v__150_59), (float)v__144);
      v__178_29 = mult2(vload2((16 + ((7 * v_N_0) / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_29, v__178_29),(16 + ((7 * v_N_0) / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_30 = mult2((float2)(v__150_60, v__150_61), (float)v__144);
      v__178_30 = mult2(vload2((32 + ((7 * v_N_0) / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_30, v__178_30),(32 + ((7 * v_N_0) / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      v__175_31 = mult2((float2)(v__150_62, v__150_63), (float)v__144);
      v__178_31 = mult2(vload2((48 + ((7 * v_N_0) / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__143), (float)v__145);
      vstore2(add2(v__175_31, v__178_31),(48 + ((7 * v_N_0) / 2) + (4 * v_N_0 * get_local_id(1)) + (32 * v_N_0 * v_wg_id_111) + (64 * v_wg_id_112) + get_local_id(0)),v__180);;
      /* end unroll */
      /* end map_seq */
      /* end unroll */
      /* end map_seq */
      /* end unroll */
      /* end unroll */
      /* end unroll */
      /* end map_seq */
    }
  }
}}


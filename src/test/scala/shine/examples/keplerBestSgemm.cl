#ifndef Tuple_float_float_DEFINED
#define Tuple_float_float_DEFINED
typedef struct {
  float _0;
  float _1;
} Tuple_float_float;
#endif

float id(float x){
  { return x; }
}
float add(float x, float y){
  { return x+y; }
}
float mult(float l, float r){
  { return l * r; }
}
kernel void keplerBestSgemm(const global float* restrict v__69, const global float* restrict v__70, const global float* restrict v__71, float v__72, float v__73, global float* v__109, int v_K_1, int v_M_0, int v_N_2){ 
/* Static local memory */
  local float v__83[1024];
  local float v__82[512];
 
#ifndef WORKGROUP_GUARD
#define WORKGROUP_GUARD
#endif
WORKGROUP_GUARD
{
  /* Typed Value memory */
  float v__77;
  /* Private Memory */
  float v__78_0;
  float v__78_1;
  float v__78_2;
  float v__78_3;
  float v__78_4;
  float v__78_5;
  float v__78_6;
  float v__78_7;
  float v__78_8;
  float v__78_9;
  float v__78_10;
  float v__78_11;
  float v__78_12;
  float v__78_13;
  float v__78_14;
  float v__78_15;
  float v__78_16;
  float v__78_17;
  float v__78_18;
  float v__78_19;
  float v__78_20;
  float v__78_21;
  float v__78_22;
  float v__78_23;
  float v__78_24;
  float v__78_25;
  float v__78_26;
  float v__78_27;
  float v__78_28;
  float v__78_29;
  float v__78_30;
  float v__78_31;
  
  float v__89_0;
  float v__89_1;
  float v__89_2;
  float v__89_3;
  float v__89_4;
  float v__89_5;
  float v__89_6;
  float v__89_7;
  
  float v__90_0;
  float v__90_1;
  float v__90_2;
  float v__90_3;
  
  float v__95_0;
  float v__95_1;
  float v__95_2;
  float v__95_3;
  float v__95_4;
  float v__95_5;
  float v__95_6;
  float v__95_7;
  float v__95_8;
  float v__95_9;
  float v__95_10;
  float v__95_11;
  float v__95_12;
  float v__95_13;
  float v__95_14;
  float v__95_15;
  float v__95_16;
  float v__95_17;
  float v__95_18;
  float v__95_19;
  float v__95_20;
  float v__95_21;
  float v__95_22;
  float v__95_23;
  float v__95_24;
  float v__95_25;
  float v__95_26;
  float v__95_27;
  float v__95_28;
  float v__95_29;
  float v__95_30;
  float v__95_31;
  
  float v__104_0;
  float v__104_1;
  float v__104_2;
  float v__104_3;
  float v__104_4;
  float v__104_5;
  float v__104_6;
  float v__104_7;
  float v__104_8;
  float v__104_9;
  float v__104_10;
  float v__104_11;
  float v__104_12;
  float v__104_13;
  float v__104_14;
  float v__104_15;
  float v__104_16;
  float v__104_17;
  float v__104_18;
  float v__104_19;
  float v__104_20;
  float v__104_21;
  float v__104_22;
  float v__104_23;
  float v__104_24;
  float v__104_25;
  float v__104_26;
  float v__104_27;
  float v__104_28;
  float v__104_29;
  float v__104_30;
  float v__104_31;
  
  float v__107_0;
  float v__107_1;
  float v__107_2;
  float v__107_3;
  float v__107_4;
  float v__107_5;
  float v__107_6;
  float v__107_7;
  float v__107_8;
  float v__107_9;
  float v__107_10;
  float v__107_11;
  float v__107_12;
  float v__107_13;
  float v__107_14;
  float v__107_15;
  float v__107_16;
  float v__107_17;
  float v__107_18;
  float v__107_19;
  float v__107_20;
  float v__107_21;
  float v__107_22;
  float v__107_23;
  float v__107_24;
  float v__107_25;
  float v__107_26;
  float v__107_27;
  float v__107_28;
  float v__107_29;
  float v__107_30;
  float v__107_31;
  
  for (int v_wg_id_42 = get_group_id(1);v_wg_id_42<(v_M_0 / (64));v_wg_id_42 = (16 + v_wg_id_42)){
    for (int v_wg_id_43 = get_group_id(0);v_wg_id_43<(v_N_2 / (128));v_wg_id_43 = (8 + v_wg_id_43)){
      float v_tmp_396 = 0.0f;
      v__77 = v_tmp_396;
      /* unroll */
      /* unroll */
      /* map_seq */
      /* unroll */
      /* map_seq */
      /* unroll */
      v__78_0 = id(v__77);
      v__78_1 = id(v__77);
      v__78_2 = id(v__77);
      v__78_3 = id(v__77);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__78_4 = id(v__77);
      v__78_5 = id(v__77);
      v__78_6 = id(v__77);
      v__78_7 = id(v__77);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__78_8 = id(v__77);
      v__78_9 = id(v__77);
      v__78_10 = id(v__77);
      v__78_11 = id(v__77);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__78_12 = id(v__77);
      v__78_13 = id(v__77);
      v__78_14 = id(v__77);
      v__78_15 = id(v__77);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__78_16 = id(v__77);
      v__78_17 = id(v__77);
      v__78_18 = id(v__77);
      v__78_19 = id(v__77);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__78_20 = id(v__77);
      v__78_21 = id(v__77);
      v__78_22 = id(v__77);
      v__78_23 = id(v__77);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__78_24 = id(v__77);
      v__78_25 = id(v__77);
      v__78_26 = id(v__77);
      v__78_27 = id(v__77);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__78_28 = id(v__77);
      v__78_29 = id(v__77);
      v__78_30 = id(v__77);
      v__78_31 = id(v__77);
      /* end unroll */
      /* end map_seq */
      /* end unroll */
      /* end map_seq */
      /* end unroll */
      /* end unroll */
      /* reduce_seq */
      for (int v_i_48 = 0;v_i_48<(v_K_1 / (8));v_i_48 = (1 + v_i_48)){
        /* iteration count is exactly 1, no loop emitted */
        {
          int v_l_id_49 = get_local_id(1);
          for (int v_l_id_52 = get_local_id(0);v_l_id_52<64;v_l_id_52 = (32 + v_l_id_52)){
            v__82[(v_l_id_52 + (64 * v_l_id_49))] = id(v__69[(v_l_id_52 + (8 * v_M_0 * v_i_48) + (v_M_0 * v_l_id_49) + (64 * v_wg_id_42))]);
          }
          barrier(CLK_LOCAL_MEM_FENCE);
          
          for (int v_l_id_53 = get_local_id(0);v_l_id_53<128;v_l_id_53 = (32 + v_l_id_53)){
            v__83[(v_l_id_53 + (128 * v_l_id_49))] = id(v__70[(v_l_id_53 + (8 * v_N_2 * v_i_48) + (v_N_2 * v_l_id_49) + (128 * v_wg_id_43))]);
          }
          barrier(CLK_LOCAL_MEM_FENCE);
          
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        
        /* unroll */
        /* unroll */
        /* reduce_seq */
        for (int v_i_56 = 0;v_i_56<8;v_i_56 = (1 + v_i_56)){
          /* map_seq */
          /* unroll */
          v__89_0 = id(v__82[((8 * get_local_id(1)) + (64 * v_i_56))]);
          v__89_1 = id(v__82[(1 + (8 * get_local_id(1)) + (64 * v_i_56))]);
          v__89_2 = id(v__82[(2 + (8 * get_local_id(1)) + (64 * v_i_56))]);
          v__89_3 = id(v__82[(3 + (8 * get_local_id(1)) + (64 * v_i_56))]);
          v__89_4 = id(v__82[(4 + (8 * get_local_id(1)) + (64 * v_i_56))]);
          v__89_5 = id(v__82[(5 + (8 * get_local_id(1)) + (64 * v_i_56))]);
          v__89_6 = id(v__82[(6 + (8 * get_local_id(1)) + (64 * v_i_56))]);
          v__89_7 = id(v__82[(7 + (8 * get_local_id(1)) + (64 * v_i_56))]);
          /* end unroll */
          /* end map_seq */
          /* map_seq */
          /* unroll */
          v__90_0 = id(v__83[((128 * v_i_56) + get_local_id(0))]);
          v__90_1 = id(v__83[(32 + (128 * v_i_56) + get_local_id(0))]);
          v__90_2 = id(v__83[(64 + (128 * v_i_56) + get_local_id(0))]);
          v__90_3 = id(v__83[(96 + (128 * v_i_56) + get_local_id(0))]);
          /* end unroll */
          /* end map_seq */
          /* map_seq */
          /* unroll */
          /* map_seq */
          /* unroll */
          v__95_0 = mult(v__89_0, v__90_0);
          v__78_0 = add(v__78_0, v__95_0);
          v__95_1 = mult(v__89_0, v__90_1);
          v__78_1 = add(v__78_1, v__95_1);
          v__95_2 = mult(v__89_0, v__90_2);
          v__78_2 = add(v__78_2, v__95_2);
          v__95_3 = mult(v__89_0, v__90_3);
          v__78_3 = add(v__78_3, v__95_3);
          /* end unroll */
          /* end map_seq */
          /* map_seq */
          /* unroll */
          v__95_4 = mult(v__89_1, v__90_0);
          v__78_4 = add(v__78_4, v__95_4);
          v__95_5 = mult(v__89_1, v__90_1);
          v__78_5 = add(v__78_5, v__95_5);
          v__95_6 = mult(v__89_1, v__90_2);
          v__78_6 = add(v__78_6, v__95_6);
          v__95_7 = mult(v__89_1, v__90_3);
          v__78_7 = add(v__78_7, v__95_7);
          /* end unroll */
          /* end map_seq */
          /* map_seq */
          /* unroll */
          v__95_8 = mult(v__89_2, v__90_0);
          v__78_8 = add(v__78_8, v__95_8);
          v__95_9 = mult(v__89_2, v__90_1);
          v__78_9 = add(v__78_9, v__95_9);
          v__95_10 = mult(v__89_2, v__90_2);
          v__78_10 = add(v__78_10, v__95_10);
          v__95_11 = mult(v__89_2, v__90_3);
          v__78_11 = add(v__78_11, v__95_11);
          /* end unroll */
          /* end map_seq */
          /* map_seq */
          /* unroll */
          v__95_12 = mult(v__89_3, v__90_0);
          v__78_12 = add(v__78_12, v__95_12);
          v__95_13 = mult(v__89_3, v__90_1);
          v__78_13 = add(v__78_13, v__95_13);
          v__95_14 = mult(v__89_3, v__90_2);
          v__78_14 = add(v__78_14, v__95_14);
          v__95_15 = mult(v__89_3, v__90_3);
          v__78_15 = add(v__78_15, v__95_15);
          /* end unroll */
          /* end map_seq */
          /* map_seq */
          /* unroll */
          v__95_16 = mult(v__89_4, v__90_0);
          v__78_16 = add(v__78_16, v__95_16);
          v__95_17 = mult(v__89_4, v__90_1);
          v__78_17 = add(v__78_17, v__95_17);
          v__95_18 = mult(v__89_4, v__90_2);
          v__78_18 = add(v__78_18, v__95_18);
          v__95_19 = mult(v__89_4, v__90_3);
          v__78_19 = add(v__78_19, v__95_19);
          /* end unroll */
          /* end map_seq */
          /* map_seq */
          /* unroll */
          v__95_20 = mult(v__89_5, v__90_0);
          v__78_20 = add(v__78_20, v__95_20);
          v__95_21 = mult(v__89_5, v__90_1);
          v__78_21 = add(v__78_21, v__95_21);
          v__95_22 = mult(v__89_5, v__90_2);
          v__78_22 = add(v__78_22, v__95_22);
          v__95_23 = mult(v__89_5, v__90_3);
          v__78_23 = add(v__78_23, v__95_23);
          /* end unroll */
          /* end map_seq */
          /* map_seq */
          /* unroll */
          v__95_24 = mult(v__89_6, v__90_0);
          v__78_24 = add(v__78_24, v__95_24);
          v__95_25 = mult(v__89_6, v__90_1);
          v__78_25 = add(v__78_25, v__95_25);
          v__95_26 = mult(v__89_6, v__90_2);
          v__78_26 = add(v__78_26, v__95_26);
          v__95_27 = mult(v__89_6, v__90_3);
          v__78_27 = add(v__78_27, v__95_27);
          /* end unroll */
          /* end map_seq */
          /* map_seq */
          /* unroll */
          v__95_28 = mult(v__89_7, v__90_0);
          v__78_28 = add(v__78_28, v__95_28);
          v__95_29 = mult(v__89_7, v__90_1);
          v__78_29 = add(v__78_29, v__95_29);
          v__95_30 = mult(v__89_7, v__90_2);
          v__78_30 = add(v__78_30, v__95_30);
          v__95_31 = mult(v__89_7, v__90_3);
          v__78_31 = add(v__78_31, v__95_31);
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
      v__104_0 = mult(v__78_0, v__72);
      v__107_0 = mult(v__71[((64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[((64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_0, v__107_0);
      v__104_1 = mult(v__78_1, v__72);
      v__107_1 = mult(v__71[(32 + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(32 + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_1, v__107_1);
      v__104_2 = mult(v__78_2, v__72);
      v__107_2 = mult(v__71[(64 + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(64 + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_2, v__107_2);
      v__104_3 = mult(v__78_3, v__72);
      v__107_3 = mult(v__71[(96 + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(96 + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_3, v__107_3);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__104_4 = mult(v__78_4, v__72);
      v__107_4 = mult(v__71[(v_N_2 + (8 * v_N_2 * get_local_id(1)) + (64 * v_N_2 * v_wg_id_42) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(v_N_2 + (8 * v_N_2 * get_local_id(1)) + (64 * v_N_2 * v_wg_id_42) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_4, v__107_4);
      v__104_5 = mult(v__78_5, v__72);
      v__107_5 = mult(v__71[(32 + v_N_2 + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(32 + v_N_2 + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_5, v__107_5);
      v__104_6 = mult(v__78_6, v__72);
      v__107_6 = mult(v__71[(64 + v_N_2 + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(64 + v_N_2 + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_6, v__107_6);
      v__104_7 = mult(v__78_7, v__72);
      v__107_7 = mult(v__71[(96 + v_N_2 + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(96 + v_N_2 + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_7, v__107_7);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__104_8 = mult(v__78_8, v__72);
      v__107_8 = mult(v__71[((2 * v_N_2) + (8 * v_N_2 * get_local_id(1)) + (64 * v_N_2 * v_wg_id_42) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[((2 * v_N_2) + (8 * v_N_2 * get_local_id(1)) + (64 * v_N_2 * v_wg_id_42) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_8, v__107_8);
      v__104_9 = mult(v__78_9, v__72);
      v__107_9 = mult(v__71[(32 + (2 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(32 + (2 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_9, v__107_9);
      v__104_10 = mult(v__78_10, v__72);
      v__107_10 = mult(v__71[(64 + (2 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(64 + (2 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_10, v__107_10);
      v__104_11 = mult(v__78_11, v__72);
      v__107_11 = mult(v__71[(96 + (2 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(96 + (2 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_11, v__107_11);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__104_12 = mult(v__78_12, v__72);
      v__107_12 = mult(v__71[((3 * v_N_2) + (8 * v_N_2 * get_local_id(1)) + (64 * v_N_2 * v_wg_id_42) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[((3 * v_N_2) + (8 * v_N_2 * get_local_id(1)) + (64 * v_N_2 * v_wg_id_42) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_12, v__107_12);
      v__104_13 = mult(v__78_13, v__72);
      v__107_13 = mult(v__71[(32 + (3 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(32 + (3 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_13, v__107_13);
      v__104_14 = mult(v__78_14, v__72);
      v__107_14 = mult(v__71[(64 + (3 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(64 + (3 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_14, v__107_14);
      v__104_15 = mult(v__78_15, v__72);
      v__107_15 = mult(v__71[(96 + (3 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(96 + (3 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_15, v__107_15);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__104_16 = mult(v__78_16, v__72);
      v__107_16 = mult(v__71[((4 * v_N_2) + (8 * v_N_2 * get_local_id(1)) + (64 * v_N_2 * v_wg_id_42) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[((4 * v_N_2) + (8 * v_N_2 * get_local_id(1)) + (64 * v_N_2 * v_wg_id_42) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_16, v__107_16);
      v__104_17 = mult(v__78_17, v__72);
      v__107_17 = mult(v__71[(32 + (4 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(32 + (4 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_17, v__107_17);
      v__104_18 = mult(v__78_18, v__72);
      v__107_18 = mult(v__71[(64 + (4 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(64 + (4 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_18, v__107_18);
      v__104_19 = mult(v__78_19, v__72);
      v__107_19 = mult(v__71[(96 + (4 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(96 + (4 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_19, v__107_19);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__104_20 = mult(v__78_20, v__72);
      v__107_20 = mult(v__71[((5 * v_N_2) + (8 * v_N_2 * get_local_id(1)) + (64 * v_N_2 * v_wg_id_42) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[((5 * v_N_2) + (8 * v_N_2 * get_local_id(1)) + (64 * v_N_2 * v_wg_id_42) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_20, v__107_20);
      v__104_21 = mult(v__78_21, v__72);
      v__107_21 = mult(v__71[(32 + (5 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(32 + (5 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_21, v__107_21);
      v__104_22 = mult(v__78_22, v__72);
      v__107_22 = mult(v__71[(64 + (5 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(64 + (5 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_22, v__107_22);
      v__104_23 = mult(v__78_23, v__72);
      v__107_23 = mult(v__71[(96 + (5 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(96 + (5 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_23, v__107_23);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__104_24 = mult(v__78_24, v__72);
      v__107_24 = mult(v__71[((6 * v_N_2) + (8 * v_N_2 * get_local_id(1)) + (64 * v_N_2 * v_wg_id_42) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[((6 * v_N_2) + (8 * v_N_2 * get_local_id(1)) + (64 * v_N_2 * v_wg_id_42) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_24, v__107_24);
      v__104_25 = mult(v__78_25, v__72);
      v__107_25 = mult(v__71[(32 + (6 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(32 + (6 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_25, v__107_25);
      v__104_26 = mult(v__78_26, v__72);
      v__107_26 = mult(v__71[(64 + (6 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(64 + (6 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_26, v__107_26);
      v__104_27 = mult(v__78_27, v__72);
      v__107_27 = mult(v__71[(96 + (6 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(96 + (6 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_27, v__107_27);
      /* end unroll */
      /* end map_seq */
      /* map_seq */
      /* unroll */
      v__104_28 = mult(v__78_28, v__72);
      v__107_28 = mult(v__71[((7 * v_N_2) + (8 * v_N_2 * get_local_id(1)) + (64 * v_N_2 * v_wg_id_42) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[((7 * v_N_2) + (8 * v_N_2 * get_local_id(1)) + (64 * v_N_2 * v_wg_id_42) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_28, v__107_28);
      v__104_29 = mult(v__78_29, v__72);
      v__107_29 = mult(v__71[(32 + (7 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(32 + (7 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_29, v__107_29);
      v__104_30 = mult(v__78_30, v__72);
      v__107_30 = mult(v__71[(64 + (7 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(64 + (7 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_30, v__107_30);
      v__104_31 = mult(v__78_31, v__72);
      v__107_31 = mult(v__71[(96 + (7 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))], v__73);
      v__109[(96 + (7 * v_N_2) + (64 * v_N_2 * v_wg_id_42) + (8 * v_N_2 * get_local_id(1)) + (128 * v_wg_id_43) + get_local_id(0))] = add(v__104_31, v__107_31);
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


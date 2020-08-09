package apps

import rise.cuda.DSL._
import rise.OpenCL.DSL._
import rise.core.DSL._
import rise.core.TypeLevelDSL._
import rise.core.types._
import util.gen

class ReduceTest extends shine.test_util.Tests {
  test("test reduce kernel"){
    val testKernel = {
      val n = 8192
      val k = 2048
      val j = 128
      val redop = add

      val id = fun(x => x)

      // idx(0) für f32 als Rückgabe wäre schöner, da kein join nach reduceWarp erforderlich
      // für Implementierung mit shfl wäre dies sowieso erforderlich, da sonst 32.f32 als Rückgabe
      // funktioniert jedoch nicht
      val reduceWarp = fun((32 `.` f32) ->: (1 `.` f32))(arr =>
        arr |>
          split(32) |>
          mapLane('x')(fun(result =>
            result |>
              oclReduceSeq(AddressSpace.Private)(redop)(l(0f))
          ))
      )

      fun(n `.` f32)(arr =>
        arr |> split(k) |> // n/k.k.f32
          mapBlock('x')(fun(chunk =>
            chunk |> split(j) |> // k/j.j.f32
              mapWarp('x')(fun(warpChunk =>
                warpChunk |> split(j/32) |> // 32.j/32.f32
                  mapLane(fun(threadChunk =>
                    threadChunk |>
                      oclReduceSeq(AddressSpace.Private)(redop)(l(0f)))) |> // 32.f32
                  toLocal |> // eigentlich toPrivate wegen shfl
                  reduceWarp)) |> join |> // k/j.f32
              toLocal |>
              padCst(0)(32-(k/j)/32)(l(0f)) |>
              split(32) |>
              mapWarp(fun(warpChunk =>
                warpChunk |>
                  mapLane(id) |>
                  toLocal |> // eigentlich toPrivate wegen shfl
                  reduceWarp)) |> join )))
    }

    gen.cuKernel(testKernel, "reduceTest")
  }
}

//  Generierter Code:
//  extern "C" __global__
//  void reduceTest(float* __restrict__ output, const float* __restrict__ x0, __shared__ float* __restrict__ x213, __shared__ float* __restrict__ x264, __shared__ float* __restrict__ x243){
//    /* Start of moved local vars */
//    /* End of moved local vars */
//    /* mapBlock */
//    for (int block_id_374 = blockIdx.x;(block_id_374 < 4);block_id_374 = (block_id_374 + gridDim.x)) {
//      /* mapWarp */
//      for (int warp_id_377 = (threadIdx.x / 32);(warp_id_377 < 16);warp_id_377 = (warp_id_377 + (blockDim.x / 32))) {
//        /* mapLane */
//        /* iteration count is exactly 1, no loop emitted */
//        int lane_id_381 = (threadIdx.x % 32);
//        /* oclReduceSeq */
//        {
//          float x276;
//          x276 = 0.0f;
//          for (int i_383 = 0;(i_383 < 4);i_383 = (1 + i_383)) {
//          x276 = (x276 + x0[(((i_383 + (4 * lane_id_381)) + (128 * warp_id_377)) + (2048 * block_id_374))]);
//        }
//
//          x264[lane_id_381] = x276;
//        }
//
//        /* mapLane */
//        for (int lane_id_382 = (threadIdx.x % 32);(lane_id_382 < 1);lane_id_382 = (32 + lane_id_382)) {
//          /* oclReduceSeq */
//          {
//            float x257;
//            x257 = 0.0f;
//            for (int i_384 = 0;(i_384 < 32);i_384 = (1 + i_384)) {
//            x257 = (x257 + x264[(i_384 + (32 * lane_id_382))]);
//          }
//
//            x243[(lane_id_382 + warp_id_377)] = x257;
//          }
//
//        }
//
//      }
//
//      /* mapWarp */
//      for (int warp_id_380 = (threadIdx.x / 32);(warp_id_380 < (6144 / 4096));warp_id_380 = (warp_id_380 + (blockDim.x / 32))) {
//        /* mapLane */
//        /* iteration count is exactly 1, no loop emitted */
//        int lane_id_385 = (threadIdx.x % 32);
//        x213[lane_id_385] = ((((lane_id_385 + (32 * warp_id_380)) < 16)) ? (x243[(lane_id_385 + (32 * warp_id_380))]) : (0.0f));
//        /* mapLane */
//        for (int lane_id_386 = (threadIdx.x % 32);(lane_id_386 < 1);lane_id_386 = (32 + lane_id_386)) {
//          /* oclReduceSeq */
//          {
//            float x206;
//            x206 = 0.0f;
//            for (int i_387 = 0;(i_387 < 32);i_387 = (1 + i_387)) {
//            x206 = (x206 + x213[(i_387 + (32 * lane_id_386))]);
//          }
//
//            output[((lane_id_386 + warp_id_380) + ((6144 * block_id_374) / 4096))] = x206;
//          }
//
//        }
//
//      }
//
//    }
//
//    __syncthreads();
//  }
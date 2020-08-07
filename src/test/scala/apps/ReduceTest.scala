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

      val reduceWarp = fun((32 `.` f32) ->: f32)(arr =>
        arr |>
          //      mapThreads(fun(threadChunk =>
          //        threadChunk |>
          oclReduceSeq(AddressSpace.Private)(redop)(l(0f))
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
                  toLocal |> //eigentlich toPrivate, da reduceWarp mit Shfl laufen soll
                  reduceWarp)) |> // k/j.f32
              toLocal |>
              padCst(0)(32-(k/j)/32)(l(0f)) |>
              split(32) |>
              mapWarp(fun(warpChunk =>
                warpChunk |>
                  mapLane(id) |>
                  toLocal |>
                  reduceWarp)))))
    }

    gen.cuKernel(testKernel, "reduceTest")
  }

  //Generierter Code:
  //extern "C" __global__
  //void reduceTest(float* __restrict__ output, const float* __restrict__ x0, __shared__ float* __restrict__ x166, __shared__ float* __restrict__ x204, __shared__ float* __restrict__ x190){
  //  /* Start of moved local vars */
  //  /* End of moved local vars */
  //  /* mapBlock */
  //  for (int block_id_289 = blockIdx.x;(block_id_289 < 4);block_id_289 = (block_id_289 + gridDim.x)) {
  //    /* mapWarp */
  //    for (int warp_id_291 = (threadIdx.x / 32);(warp_id_291 < 16);warp_id_291 = (warp_id_291 + (blockDim.x / 32))) {
  //      /* mapLane */
  //      /* iteration count is exactly 1, no loop emitted */
  //      int lane_id_294 = (threadIdx.x % 32);
  //      /* oclReduceSeq */
  //      {
  //        float x216;
  //        x216 = 0.0f;
  //        for (int i_295 = 0;(i_295 < 4);i_295 = (1 + i_295)) {
  //          x216 = (x216 + x0[(((i_295 + (4 * lane_id_294)) + (128 * warp_id_291)) + (2048 * block_id_289))]);
  //        }
  //
  //        x204[lane_id_294] = x216;
  //      }
  //
  //      /* oclReduceSeq */
  //      {
  //        float x202;
  //        x202 = 0.0f;
  //        for (int i_296 = 0;(i_296 < 32);i_296 = (1 + i_296)) {
  //          x202 = (x202 + x204[i_296]);
  //        }
  //
  //        x190[warp_id_291] = x202;
  //      }
  //
  //    }
  //
  //    /* mapWarp */
  //    for (int warp_id_293 = (threadIdx.x / 32);(warp_id_293 < (6144 / 4096));warp_id_293 = (warp_id_293 + (blockDim.x / 32))) {
  //      /* mapLane */
  //      /* iteration count is exactly 1, no loop emitted */
  //      int lane_id_297 = (threadIdx.x % 32);
  //      x166[lane_id_297] = ((((lane_id_297 + (32 * warp_id_293)) < 16)) ? (x190[(lane_id_297 + (32 * warp_id_293))]) : (0.0f));
  //      /* oclReduceSeq */
  //      {
  //        float x164;
  //        x164 = 0.0f;
  //        for (int i_298 = 0;(i_298 < 32);i_298 = (1 + i_298)) {
  //          x164 = (x164 + x166[i_298]);
  //        }
  //
  //        output[(warp_id_293 + ((6144 * block_id_289) / 4096))] = x164;
  //      }
  //
  //    }
  //
  //  }
  //
  //  __syncthreads();
  //}
}
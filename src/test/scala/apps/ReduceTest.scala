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

      // max block size in cuda ist 1024 threads (also 32 warps)

      // idx(0) für f32 als Rückgabe ist schöner, da kein join nach reduceWarp erforderlich
      // für Implementierung mit shfl wäre dies sowieso erforderlich, da sonst 32.f32 als Rückgabe
      // funktioniert jedoch nicht
      // Fehlermeldung:
      //  rise.core.types.InferenceException was thrown.
      //  inference exception: expected a dependent function type, but got (idx[_n18] -> (_n18._dt19 -> _dt19))
      val reduceWarp = fun((32 `.` f32) ->: f32)(arr =>
        arr |>
          split(32) |>
          mapLane('x')(fun(result =>
            result |>
              oclReduceSeq(AddressSpace.Private)(redop)(l(0f))
          )) |> idx(0)
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
                  toLocal |> // eigentlich im private memory lassen wegen shfl
                  reduceWarp)) |> // k/j.f32
              toLocal |>
              padCst(0)(32-(k/j))(l(0f)) |> // padde um 32-#Warps viele Elemente um auf 32 zu kommen
              split(32) |>
              mapWarp(fun(warpChunk =>
                warpChunk |>
                  mapLane(id) |>
                  toLocal |> // eigentlich toPrivate wegen shfl
                  reduceWarp)))))
    }

    gen.cuKernel(testKernel, "reduceTest")
  }
}

//  Generierter Code: (ohne idx(0))
//extern "C" __global__
//void reduceTest(float* __restrict__ output, const float* __restrict__ x0, __shared__ float* __restrict__ x215, __shared__ float* __restrict__ x266, __shared__ float* __restrict__ x245){
//  /* Start of moved local vars */
//  /* End of moved local vars */
//  /* mapBlock */
//  for (int block_id_376 = blockIdx.x;(block_id_376 < 4);block_id_376 = (block_id_376 + gridDim.x)) {
//  /* mapWarp */
//  for (int warp_id_379 = (threadIdx.x / 32);(warp_id_379 < 16);warp_id_379 = (warp_id_379 + (blockDim.x / 32))) {
//  /* mapLane */
//  /* iteration count is exactly 1, no loop emitted */
//  int lane_id_383 = (threadIdx.x % 32);
//  /* oclReduceSeq */
//{
//  float x278;
//  x278 = 0.0f;
//  for (int i_385 = 0;(i_385 < 4);i_385 = (1 + i_385)) {
//  x278 = (x278 + x0[(((i_385 + (4 * lane_id_383)) + (128 * warp_id_379)) + (2048 * block_id_376))]);
//}
//
//  x266[lane_id_383] = x278;
//}
//
//  /* mapLane */
//  for (int lane_id_384 = (threadIdx.x % 32);(lane_id_384 < 1);lane_id_384 = (32 + lane_id_384)) {
//  /* oclReduceSeq */
//{
//  float x259;
//  x259 = 0.0f;
//  for (int i_386 = 0;(i_386 < 32);i_386 = (1 + i_386)) {
//  x259 = (x259 + x266[(i_386 + (32 * lane_id_384))]);
//}
//
//  x245[(lane_id_384 + warp_id_379)] = x259;
//}
//
//}
//
//}
//
//  /* mapWarp */
//  for (int warp_id_382 = (threadIdx.x / 32);(warp_id_382 < 1);warp_id_382 = (warp_id_382 + (blockDim.x / 32))) {
//  /* mapLane */
//  /* iteration count is exactly 1, no loop emitted */
//  int lane_id_387 = (threadIdx.x % 32);
//  x215[lane_id_387] = ((((lane_id_387 + (32 * warp_id_382)) < 16)) ? (x245[(lane_id_387 + (32 * warp_id_382))]) : (0.0f));
//  /* mapLane */
//  for (int lane_id_388 = (threadIdx.x % 32);(lane_id_388 < 1);lane_id_388 = (32 + lane_id_388)) {
//  /* oclReduceSeq */
//{
//  float x208;
//  x208 = 0.0f;
//  for (int i_389 = 0;(i_389 < 32);i_389 = (1 + i_389)) {
//  x208 = (x208 + x215[(i_389 + (32 * lane_id_388))]);
//}
//
//  output[((block_id_376 + lane_id_388) + warp_id_382)] = x208;
//}
//
//}
//
//}
//
//}
//
//  __syncthreads();
//}
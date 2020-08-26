package apps

import rise.cuda.DSL._
import rise.OpenCL.DSL._
import rise.core.DSL._
import rise.core.Expr
import rise.core.TypeLevelDSL._
import rise.core.types._
import util.gen

class ReduceTest extends shine.test_util.Tests {
  test("block reduce test"){
    val blockTest = {

      val op = fun(f32 x f32)(t => t._1 + t._2)

      // reduceBlock: (n: nat) -> n.f32 -> n/32.f32, where n % 32 = 0
      nFun(n => fun(n `.` f32)(arr =>
        arr |> //32.f32
          split(warpSize) |>
          mapWarp(warpReduceFinal(op)) |> //n/32.1.f32
          join // n/32.f32, where n/32 == #warps
      ))
    }
    gen.cuKernel(blockTest, "blockReduceTest")
  }

  val warpSize = 32
  private val id = fun(x => x)
  private def warpReduceIntermediate(op: Expr): Expr = {
    fun(warpChunk =>
      warpChunk |>
        toPrivateFun(mapLane(fun(threadChunk =>
          threadChunk |>
            oclReduceSeq(AddressSpace.Private)(add)(l(0f))
        ))) |> //32.f32
        let(fun(x => zip(x, x))) |> //32.(f32 x f32)
        toPrivateFun(mapLane(op)) |> //32.f32
        let(fun(x => zip(x, x))) |> //32.(f32 x f32)
        toPrivateFun(mapLane(op)) |> //32.f32
        let(fun(x => zip(x, x))) |> //32.(f32 x f32)
        toPrivateFun(mapLane(op)) |> //32.f32
        let(fun(x => zip(x, x))) |> //32.(f32 x f32)
        toPrivateFun(mapLane(op)) |> //32.f32
        let(fun(x => zip(x, x))) |> //32.(f32 x f32)
        toPrivateFun(mapLane(op)) |> //32.f32
        take(1) |>
        mapLane(id)
    )
  }

  private def warpReduceFinal(op: Expr): Expr = {
    fun(warpChunk =>
      warpChunk |>
        toPrivateFun(mapLane(id)) |> //32.f32
        let(fun(x => zip(x, x))) |> //32.(f32 x f32)
        toPrivateFun(mapLane(op)) |> //32.f32
        let(fun(x => zip(x, x))) |> //32.(f32 x f32)
        toPrivateFun(mapLane(op)) |> //32.f32
        let(fun(x => zip(x, x))) |> //32.(f32 x f32)
        toPrivateFun(mapLane(op)) |> //32.f32
        let(fun(x => zip(x, x))) |> //32.(f32 x f32)
        toPrivateFun(mapLane(op)) |> //32.f32
        let(fun(x => zip(x, x))) |> //32.(f32 x f32)
        toPrivateFun(mapLane(op)) |> //32.f32
        // Take the SUBarray consisting of 1 element(s) starting from
        // the beginning of the array.
        take(1) |> //1.f32
        // Map id over the array to say that the copying the result out is a lane specific
        // and not warp specific operation.
        // We cannot retunr f32 with idx, because this access would be executed by the entire warp.
        // Idx needs to access an array, so mapLane needs to write into memory first (single element array)
        // and then we need to copy with idx from that single element array. But the element returned from idx is
        // then copied by the entire warp again!
        // So instead, we need to just return the single element arrays.
        mapLane(id) //1.f32
    )
  }

  test("device reduce test"){
    val deviceTest = {

      val op = fun(f32 x f32)(t => t._1 + t._2)

      // reduceDevice: (n: nat) -> n.f32 -> n/32.f32, where n % 32 = 0
      nFun(n =>
        nFun(numElemsBlock =>
          nFun(numElemsWarp => fun(n `.` f32)(arr =>
           arr |> split(numElemsBlock) |> // n/numElemsBlock.numElemsBlock.f32
            mapBlock('x')(fun(chunk =>
              chunk |> split(numElemsWarp) |> // numElemsBlock/numElemsWarp.numElemsWarp.f32
                mapWarp('x')(fun(warpChunk =>
                  warpChunk |> split(numElemsWarp/warpSize) |>
                    warpReduceIntermediate(op)
                ))
                |> toLocal |> //(k/j).1.f32 where (k/j) = #warps per block
                join |> //(k/j).f32 where (k/j) = #warps per block
                padCst(0)(warpSize-(numElemsBlock/numElemsWarp))(l(0f)) |> //32.f32
                split(warpSize) |> //1.32.f32 in order to execute following reduction with one warp
                mapWarp(warpReduceFinal(op)) |>
                join
            ))
          ))
        ))

    }
    gen.cuKernel(deviceTest, "deviceReduceTest")
  }

  //deprecated
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
      val reduceWarp = fun((32 `.` f32) ->: (1 `.` f32))(arr =>
        arr |>
          split(32) |>
          mapLane('x')(fun(result =>
            result |>
              oclReduceSeq(AddressSpace.Private)(redop)(l(0f))
          )) //|> idx(0)
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
                  // toLocal |> // eigentlich im private memory lassen wegen shfl
                  reduceWarp)) |> join |> // k/j.f32
              toLocal |>
              padCst(0)(32-(k/j))(l(0f)) |> // padde um 32-#Warps viele Elemente um auf 32 zu kommen
              split(32) |>
              mapWarp(fun(warpChunk =>
                warpChunk |>
                  //mapLane(id) |> // für toLocal erforderlich
                  toPrivateFun(mapLane(id)) |> // eigentlich toPrivate wegen shfl
                  reduceWarp)) |> join )))
    }

    gen.cuKernel(testKernel, "reduceTest")
  }
}

//  Generierter Code: (ohne idx(0))
 /*
extern "C" __global__
void reduceTest(float* __restrict__ output, const float* __restrict__ x0, __shared__ float* __restrict__ x333, __shared__ float* __restrict__ x384, __shared__ float* __restrict__ x363){
  /* Start of moved local vars */
  /* End of moved local vars */
  /* mapBlock */
  for (int block_id_494 = blockIdx.x;(block_id_494 < 4);block_id_494 = (block_id_494 + gridDim.x)) {
    /* mapWarp */
    for (int warp_id_497 = (threadIdx.x / 32);(warp_id_497 < 16);warp_id_497 = (warp_id_497 + (blockDim.x / 32))) {
      /* mapLane */
      /* iteration count is exactly 1, no loop emitted */
      int lane_id_501 = (threadIdx.x % 32);
      /* oclReduceSeq */
      {
        float x396;
        x396 = 0.0f;
        for (int i_503 = 0;(i_503 < 4);i_503 = (1 + i_503)) {
          x396 = (x396 + x0[(((i_503 + (4 * lane_id_501)) + (128 * warp_id_497)) + (2048 * block_id_494))]);
        }

        x384[lane_id_501] = x396;
      }

      /* mapLane */
      for (int lane_id_502 = (threadIdx.x % 32);(lane_id_502 < 1);lane_id_502 = (32 + lane_id_502)) {
        /* oclReduceSeq */
        {
          float x377;
          x377 = 0.0f;
          for (int i_504 = 0;(i_504 < 32);i_504 = (1 + i_504)) {
            x377 = (x377 + x384[(i_504 + (32 * lane_id_502))]);
          }

          x363[(lane_id_502 + warp_id_497)] = x377;
        }

      }

    }

    /* mapWarp */
    for (int warp_id_500 = (threadIdx.x / 32);(warp_id_500 < 1);warp_id_500 = (warp_id_500 + (blockDim.x / 32))) {
      /* mapLane */
      /* iteration count is exactly 1, no loop emitted */
      int lane_id_505 = (threadIdx.x % 32);
      x333[lane_id_505] = ((((lane_id_505 + (32 * warp_id_500)) < 16)) ? (x363[(lane_id_505 + (32 * warp_id_500))]) : (0.0f));
      /* mapLane */
      for (int lane_id_506 = (threadIdx.x % 32);(lane_id_506 < 1);lane_id_506 = (32 + lane_id_506)) {
        /* oclReduceSeq */
        {
          float x326;
          x326 = 0.0f;
          for (int i_507 = 0;(i_507 < 32);i_507 = (1 + i_507)) {
            x326 = (x326 + x333[(i_507 + (32 * lane_id_506))]);
          }

          output[((block_id_494 + lane_id_506) + warp_id_500)] = x326;
        }

      }

    }

  }

  __syncthreads();
}
 */
package apps

import rise.cuda.DSL._
import rise.OpenCL.DSL.{oclReduceSeq, toLocal, toPrivate, toPrivateFun}
import rise.core.DSL._
import rise.core._
import rise.core.TypeLevelDSL._
import rise.core.types._
import util.gen

class ReduceTest extends shine.test_util.Tests {

  val warpSize = 32
  val srcLanes = generate(fun(IndexType(warpSize))(i => i ))

  private val id = fun(x => x)
  private def warpReduce(op: Expr): Expr = {
    fun(warpChunk =>
      warpChunk |>
        toPrivateFun(mapLane(id)) |> //32.f32
        let(fun(x => zip(x, x |> shflDownWarp(16)))) |> //32.(f32 x f32)
        toPrivateFun(mapLane(op)) |> //32.f32
        let(fun(x => zip(x, x |> shflDownWarp(8)))) |> //32.(f32 x f32)
        toPrivateFun(mapLane(op)) |> //32.f32
        let(fun(x => zip(x, x |> shflDownWarp(4)))) |> //32.(f32 x f32)
        toPrivateFun(mapLane(op)) |> //32.f32
        let(fun(x => zip(x, x |> shflDownWarp(2)))) |> //32.(f32 x f32)
        toPrivateFun(mapLane(op)) |> //32.f32
        let(fun(x => zip(x, x |> shflDownWarp(1)))) |> //32.(f32 x f32)
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

  private def blockReduce(op: Expr): Expr = {
    fun(blockChunk =>
      blockChunk |> split(warpSize) |>
        mapWarp('x')(fun(warpChunk =>
          warpChunk |>
            mapLane('x')(fun(laneVal =>
              laneVal |> id
            )) |>
            toPrivate |>
            warpReduce(op)
        )) |>
          toLocal |>
          join |>
          //padCst(0)(warpSize-(blockDim /^ warpSize))(l(0f)) |>
          split(warpSize) |>
          mapWarp(warpReduce(op)) |>
          join
    )
  }

  test("device reduce test2"){
    val deviceTest = {

      val op = fun(f32 x f32)(t => t._1 + t._2)

      // reduceDevice: (n: nat) -> n.f32 -> n/numElemsBlock.f32, where n % numElemsBlock = 0
      nFun((n, elemsSeq) =>
          fun(n `.` f32)(arr =>
            arr |> split(elemsSeq) |>
              mapGlobal('x')(fun(seqChunk =>
                seqChunk |>
                  oclReduceSeq(AddressSpace.Private)(add)(l(0f))
              )) |>
              padCst(0)(1023-n%1024)(l(0f)) |>
              split(1024) |> // n/numElemsBlock.numElemsBlock.f32
              mapBlock('x')(blockReduce(op))
          )
      )
    }
    //64 blocks, 32 warps each; 32 elems per lane
    gen.cuKernel(deviceTest(2097152)(32), "deviceReduceGenerated")
  }

  test("device reduce test"){
    val deviceTest = {

      val op = fun(f32 x f32)(t => t._1 + t._2)

      // reduceDevice: (n: nat) -> n.f32 -> n/numElemsBlock.f32, where n % numElemsBlock = 0
      nFun((n, numElemsBlock, numElemsWarp) =>
        fun(n `.` f32)(arr =>
          arr |> split(numElemsBlock) |> // n/numElemsBlock.numElemsBlock.f32
            mapBlock('x')(fun(chunk =>
              chunk |> split(numElemsWarp) |> // numElemsBlock/numElemsWarp.numElemsWarp.f32
                mapWarp('x')(fun(warpChunk =>
                  warpChunk |> split(numElemsWarp /^ warpSize) |> // warpSize.numElemsWarp/warpSize.f32
                    mapLane(fun(threadChunk =>
                      threadChunk |>
                        oclReduceSeq(AddressSpace.Private)(add)(l(0f))
                    )) |> toPrivate |>
                    warpReduce(op))) |> toLocal |> //(k/j).1.f32 where (k/j) = #warps per block
                join |> //(k/j).f32 where (k/j) = #warps per block
                padCst(0)(warpSize-(numElemsBlock /^ numElemsWarp))(l(0f)) |> //32.f32
                split(warpSize) |> //1.32.f32 in order to execute following reduction with one warp
                mapWarp(warpReduce(op)) |>
                join
            ))
        ))
    }
    //64 blocks, 32 warps each; 32 elems per lane
    gen.cuKernel(deviceTest(2097152)(32768)(2048), "deviceReduceGenerated")
  }

  // Generierter Code:
  /*
extern "C" __global__
void deviceReduceGenerated(float* __restrict__ output, const float* __restrict__ x0){
  extern __shared__ char dynamicSharedMemory1417[];
  /* mapBlock */
  for (int block_id_1304 = blockIdx.x;(block_id_1304 < 64);block_id_1304 = (block_id_1304 + gridDim.x)) {
    {
      float* x906 = ((float*)(&(dynamicSharedMemory1417[0])));
      /* mapWarp */
      for (int warp_id_1330 = (threadIdx.x / 32);(warp_id_1330 < 16);warp_id_1330 = (warp_id_1330 + (blockDim.x / 32))) {
        {
          float x919[1];
          {
            float x939[1];
            {
              float x959[1];
              {
                float x979[1];
                {
                  float x999[1];
                  {
                    float x1019[1];
                    {
                      float x1027[1];
                      /* mapLane */
                      /* iteration count is exactly 1, no loop emitted */
                      int lane_id_1380 = (threadIdx.x % 32);
                      /* oclReduceSeq */
                      {
                        float x1039;
                        x1039 = 0.0f;
                        for (int i_1382 = 0;(i_1382 < 64);i_1382 = (1 + i_1382)) {
                          x1039 = (x1039 + x0[(((i_1382 + (64 * lane_id_1380)) + (2048 * warp_id_1330)) + (32768 * block_id_1304))]);
                        }

                        x1027[0] = x1039;
                      }

                      __syncwarp();
                      /* mapLane */
                      /* iteration count is exactly 1, no loop emitted */
                      int lane_id_1381 = (threadIdx.x % 32);
                      x1019[0] = x1027[0];
                      __syncwarp();
                    }

                    /* mapLane */
                    /* iteration count is exactly 1, no loop emitted */
                    int lane_id_1379 = (threadIdx.x % 32);
                    x999[0] = (x1019[0] + __shfl_down_sync(0xFFFFFFFF, x1019[0], 16));
                    __syncwarp();
                  }

                  /* mapLane */
                  /* iteration count is exactly 1, no loop emitted */
                  int lane_id_1376 = (threadIdx.x % 32);
                  x979[0] = (x999[0] + __shfl_down_sync(0xFFFFFFFF, x999[0], 8));
                  __syncwarp();
                }

                /* mapLane */
                /* iteration count is exactly 1, no loop emitted */
                int lane_id_1372 = (threadIdx.x % 32);
                x959[0] = (x979[0] + __shfl_down_sync(0xFFFFFFFF, x979[0], 4));
                __syncwarp();
              }

              /* mapLane */
              /* iteration count is exactly 1, no loop emitted */
              int lane_id_1367 = (threadIdx.x % 32);
              x939[0] = (x959[0] + __shfl_down_sync(0xFFFFFFFF, x959[0], 2));
              __syncwarp();
            }

            /* mapLane */
            /* iteration count is exactly 1, no loop emitted */
            int lane_id_1361 = (threadIdx.x % 32);
            x919[0] = (x939[0] + __shfl_down_sync(0xFFFFFFFF, x939[0], 1));
            __syncwarp();
          }

          /* mapLane */
          for (int lane_id_1354 = (threadIdx.x % 32);(lane_id_1354 < 1);lane_id_1354 = (32 + lane_id_1354)) {
            x906[(lane_id_1354 + warp_id_1330)] = x919[0];
          }

          __syncwarp();
        }

      }

      __syncthreads();
      /* mapWarp */
      for (int warp_id_1338 = (threadIdx.x / 32);(warp_id_1338 < 1);warp_id_1338 = (warp_id_1338 + (blockDim.x / 32))) {
        {
          float x717[1];
          {
            float x737[1];
            {
              float x757[1];
              {
                float x777[1];
                {
                  float x797[1];
                  {
                    float x817[1];
                    /* mapLane */
                    /* iteration count is exactly 1, no loop emitted */
                    int lane_id_1415 = (threadIdx.x % 32);
                    x817[0] = ((((lane_id_1415 + (32 * warp_id_1338)) < 16)) ? (x906[(lane_id_1415 + (32 * warp_id_1338))]) : (0.0f));
                    __syncwarp();
                    /* mapLane */
                    /* iteration count is exactly 1, no loop emitted */
                    int lane_id_1416 = (threadIdx.x % 32);
                    x797[0] = (x817[0] + __shfl_down_sync(0xFFFFFFFF, x817[0], 16));
                    __syncwarp();
                  }

                  /* mapLane */
                  /* iteration count is exactly 1, no loop emitted */
                  int lane_id_1414 = (threadIdx.x % 32);
                  x777[0] = (x797[0] + __shfl_down_sync(0xFFFFFFFF, x797[0], 8));
                  __syncwarp();
                }

                /* mapLane */
                /* iteration count is exactly 1, no loop emitted */
                int lane_id_1411 = (threadIdx.x % 32);
                x757[0] = (x777[0] + __shfl_down_sync(0xFFFFFFFF, x777[0], 4));
                __syncwarp();
              }

              /* mapLane */
              /* iteration count is exactly 1, no loop emitted */
              int lane_id_1407 = (threadIdx.x % 32);
              x737[0] = (x757[0] + __shfl_down_sync(0xFFFFFFFF, x757[0], 2));
              __syncwarp();
            }

            /* mapLane */
            /* iteration count is exactly 1, no loop emitted */
            int lane_id_1402 = (threadIdx.x % 32);
            x717[0] = (x737[0] + __shfl_down_sync(0xFFFFFFFF, x737[0], 1));
            __syncwarp();
          }

          /* mapLane */
          for (int lane_id_1396 = (threadIdx.x % 32);(lane_id_1396 < 1);lane_id_1396 = (32 + lane_id_1396)) {
            output[((block_id_1304 + lane_id_1396) + warp_id_1338)] = x717[0];
          }

          __syncwarp();
        }

      }

      __syncthreads();
    }

  }

  __syncthreads();
}

   */

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
package apps

import rise.cuda.DSL._
import rise.OpenCL.DSL._
import rise.core.DSL._
import rise.core._
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
          mapWarp(warpReduce(op)) |> //n/32.1.f32
          join // n/32.f32, where n/32 == #warps
      ))
    }
    gen.cuKernel(blockTest, "blockReduceTest")
  }



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

  test("device reduce test"){
    val deviceTest = {

      val op = fun(f32 x f32)(t => t._1 + t._2)

      // reduceDevice: (n: nat) -> n.f32 -> n/numElemsBlock.f32, where n % numElemsBlock = 0
      nFun(n =>
        nFun(numElemsBlock =>
          nFun(numElemsWarp => fun(n `.` f32)(arr =>
            arr |> split(numElemsBlock) |> // n/numElemsBlock.numElemsBlock.f32
              mapBlock('x')(fun(chunk =>
                chunk |> split(numElemsWarp) |> // numElemsBlock/numElemsWarp.numElemsWarp.f32
                  mapWarp('x')(fun(warpChunk =>
                    warpChunk |> split(numElemsWarp /^ warpSize) |> // warpSize.numElemsWarp/warpSize.f32
                      //TODO: what should we call this
                      //FIXME: fuse this and warpReduce into a single mapLane
                      mapLane(fun(threadChunk =>
                        threadChunk |>
                          oclReduceSeq(AddressSpace.Private)(add)(l(0f))
                        //TODO: THIS was missing. but it creates an unnecessary copy, so the upper mapLane
                        // should be inside of mapWarp in reduceWarp.
                        // We need 2 warpReduce, like warpReduceIntermediate and warpReduceFinal (or better names)
                      )) |> toPrivate |>
                      warpReduce(op))) |> toLocal |> //(k/j).1.f32 where (k/j) = #warps per block
                  join |> //(k/j).f32 where (k/j) = #warps per block
                  padCst(0)(warpSize-(numElemsBlock /^ numElemsWarp))(l(0f)) |> //32.f32
                  split(warpSize) |> //1.32.f32 in order to execute following reduction with one warp
                  mapWarp(warpReduce(op)) |>
                  join
              ))
          ))
        ))

    }
    gen.cuKernel(deviceTest(2048)(1024)(32), "deviceReduceTest")
  }

  // Generierter Code:
  /*
extern "C" __global__
void deviceReduceTest(float* __restrict__ output, const float* __restrict__ x0, __shared__ float* __restrict__ x906){
  /* Start of moved local vars */
  /* End of moved local vars */
  /* mapBlock */
  for (int block_id_1304 = blockIdx.x;(block_id_1304 < 2);block_id_1304 = (block_id_1304 + gridDim.x)) {
    /* mapWarp */
    for (int warp_id_1313 = (threadIdx.x / 32);(warp_id_1313 < 32);warp_id_1313 = (warp_id_1313 + (blockDim.x / 32))) {
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
                    int lane_id_1363 = (threadIdx.x % 32);
                    /* oclReduceSeq */
                    {
                      float x1039;
                      x1039 = 0.0f;
                      /* iteration count is exactly 1, no loop emitted */
                      int i_1365 = 0;
                      x1039 = (x1039 + x0[((lane_id_1363 + (32 * warp_id_1313)) + (1024 * block_id_1304))]);
                      x1027[0] = x1039;
                    }

                    __syncwarp()
                    /* mapLane */
                    /* iteration count is exactly 1, no loop emitted */
                    int lane_id_1364 = (threadIdx.x % 32);
                    x1019[0] = x1027[0];
                    __syncwarp()
                  }

                  /* mapLane */
                  /* iteration count is exactly 1, no loop emitted */
                  int lane_id_1362 = (threadIdx.x % 32);
                  x999[0] = (x1019[0] + __shfl_down_sync(0xFFFFFFFF, x1019[0], n460));
                  __syncwarp()
                }

                /* mapLane */
                /* iteration count is exactly 1, no loop emitted */
                int lane_id_1359 = (threadIdx.x % 32);
                x979[0] = (x999[0] + __shfl_down_sync(0xFFFFFFFF, x999[0], n425));
                __syncwarp()
              }

              /* mapLane */
              /* iteration count is exactly 1, no loop emitted */
              int lane_id_1355 = (threadIdx.x % 32);
              x959[0] = (x979[0] + __shfl_down_sync(0xFFFFFFFF, x979[0], n390));
              __syncwarp()
            }

            /* mapLane */
            /* iteration count is exactly 1, no loop emitted */
            int lane_id_1350 = (threadIdx.x % 32);
            x939[0] = (x959[0] + __shfl_down_sync(0xFFFFFFFF, x959[0], n355));
            __syncwarp()
          }

          /* mapLane */
          /* iteration count is exactly 1, no loop emitted */
          int lane_id_1344 = (threadIdx.x % 32);
          x919[0] = (x939[0] + __shfl_down_sync(0xFFFFFFFF, x939[0], n320));
          __syncwarp()
        }

        /* mapLane */
        for (int lane_id_1337 = (threadIdx.x % 32);(lane_id_1337 < 1);lane_id_1337 = (32 + lane_id_1337)) {
          x906[(lane_id_1337 + warp_id_1313)] = x919[0];
        }

        __syncwarp()
      }

    }

    __syncthreads();
    /* mapWarp */
    for (int warp_id_1321 = (threadIdx.x / 32);(warp_id_1321 < 1);warp_id_1321 = (warp_id_1321 + (blockDim.x / 32))) {
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
                  int lane_id_1398 = (threadIdx.x % 32);
                  x817[0] = x906[(lane_id_1398 + (32 * warp_id_1321))];
                  __syncwarp()
                  /* mapLane */
                  /* iteration count is exactly 1, no loop emitted */
                  int lane_id_1399 = (threadIdx.x % 32);
                  x797[0] = (x817[0] + __shfl_down_sync(0xFFFFFFFF, x817[0], n225));
                  __syncwarp()
                }

                /* mapLane */
                /* iteration count is exactly 1, no loop emitted */
                int lane_id_1397 = (threadIdx.x % 32);
                x777[0] = (x797[0] + __shfl_down_sync(0xFFFFFFFF, x797[0], n190));
                __syncwarp()
              }

              /* mapLane */
              /* iteration count is exactly 1, no loop emitted */
              int lane_id_1394 = (threadIdx.x % 32);
              x757[0] = (x777[0] + __shfl_down_sync(0xFFFFFFFF, x777[0], n155));
              __syncwarp()
            }

            /* mapLane */
            /* iteration count is exactly 1, no loop emitted */
            int lane_id_1390 = (threadIdx.x % 32);
            x737[0] = (x757[0] + __shfl_down_sync(0xFFFFFFFF, x757[0], n120));
            __syncwarp()
          }

          /* mapLane */
          /* iteration count is exactly 1, no loop emitted */
          int lane_id_1385 = (threadIdx.x % 32);
          x717[0] = (x737[0] + __shfl_down_sync(0xFFFFFFFF, x737[0], n85));
          __syncwarp()
        }

        /* mapLane */
        for (int lane_id_1379 = (threadIdx.x % 32);(lane_id_1379 < 1);lane_id_1379 = (32 + lane_id_1379)) {
          output[((block_id_1304 + lane_id_1379) + warp_id_1321)] = x717[0];
        }

        __syncwarp()
      }

    }

    __syncthreads();
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
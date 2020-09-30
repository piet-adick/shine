package apps

import rise.cuda.DSL._
import rise.OpenCL.DSL._
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
    //64 blocks, 32 warps each; 32 elems per lane
    gen.cuKernel(deviceTest(2097152)(32768)(1024), "deviceReduceGenerated")
    gen.cuKernel(deviceTest(64)(64)(32), "deviceReduceGeneratedFinal")
  }

  // Generierter Code:
  /*
extern "C" __global__
void deviceReduceTest(float* __restrict__ output, const float* __restrict__ x0){
  extern __shared__ char dynamicSharedMemory1417[];
  /* mapBlock */
  for (int block_id_1304 = blockIdx.x;(block_id_1304 < 72);block_id_1304 = (block_id_1304 + gridDim.x)) {
    {
      float* x906 = ((float*)(&(dynamicSharedMemory1417[0])));
      /* mapWarp */
      for (int warp_id_1330 = (threadIdx.x / 32);(warp_id_1330 < 32);warp_id_1330 = (warp_id_1330 + (blockDim.x / 32))) {
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
                        for (int i_1382 = 0;(i_1382 < 32);i_1382 = (1 + i_1382)) {
                          x1039 = (x1039 + x0[(((i_1382 + (32 * lane_id_1380)) + (1024 * warp_id_1330)) + (32768 * block_id_1304))]);
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
                    x817[0] = x906[(lane_id_1415 + (32 * warp_id_1338))];
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


extern "C" __global__
void deviceReduceGeneratedFinal(float* __restrict__ output, const float* __restrict__ x0){
  extern __shared__ char dynamicSharedMemory2800[];
  /* mapBlock */
  for (int block_id_2687 = blockIdx.x;(block_id_2687 < 1);block_id_2687 = (block_id_2687 + gridDim.x)) {
    {
      float* x2289 = ((float*)(&(dynamicSharedMemory2800[0])));
      /* mapWarp */
      for (int warp_id_2713 = (threadIdx.x / 32);(warp_id_2713 < 2);warp_id_2713 = (warp_id_2713 + (blockDim.x / 32))) {
        {
          float x2302[1];
          {
            float x2322[1];
            {
              float x2342[1];
              {
                float x2362[1];
                {
                  float x2382[1];
                  {
                    float x2402[1];
                    {
                      float x2410[1];
                      /* mapLane */
                      /* iteration count is exactly 1, no loop emitted */
                      int lane_id_2763 = (threadIdx.x % 32);
                      /* oclReduceSeq */
                      {
                        float x2422;
                        x2422 = 0.0f;
                        /* iteration count is exactly 1, no loop emitted */
                        int i_2765 = 0;
                        x2422 = (x2422 + x0[((lane_id_2763 + (32 * warp_id_2713)) + (64 * block_id_2687))]);
                        x2410[0] = x2422;
                      }

                      __syncwarp();
                      /* mapLane */
                      /* iteration count is exactly 1, no loop emitted */
                      int lane_id_2764 = (threadIdx.x % 32);
                      x2402[0] = x2410[0];
                      __syncwarp();
                    }

                    /* mapLane */
                    /* iteration count is exactly 1, no loop emitted */
                    int lane_id_2762 = (threadIdx.x % 32);
                    x2382[0] = (x2402[0] + __shfl_down_sync(0xFFFFFFFF, x2402[0], 16));
                    __syncwarp();
                  }

                  /* mapLane */
                  /* iteration count is exactly 1, no loop emitted */
                  int lane_id_2759 = (threadIdx.x % 32);
                  x2362[0] = (x2382[0] + __shfl_down_sync(0xFFFFFFFF, x2382[0], 8));
                  __syncwarp();
                }

                /* mapLane */
                /* iteration count is exactly 1, no loop emitted */
                int lane_id_2755 = (threadIdx.x % 32);
                x2342[0] = (x2362[0] + __shfl_down_sync(0xFFFFFFFF, x2362[0], 4));
                __syncwarp();
              }

              /* mapLane */
              /* iteration count is exactly 1, no loop emitted */
              int lane_id_2750 = (threadIdx.x % 32);
              x2322[0] = (x2342[0] + __shfl_down_sync(0xFFFFFFFF, x2342[0], 2));
              __syncwarp();
            }

            /* mapLane */
            /* iteration count is exactly 1, no loop emitted */
            int lane_id_2744 = (threadIdx.x % 32);
            x2302[0] = (x2322[0] + __shfl_down_sync(0xFFFFFFFF, x2322[0], 1));
            __syncwarp();
          }

          /* mapLane */
          for (int lane_id_2737 = (threadIdx.x % 32);(lane_id_2737 < 1);lane_id_2737 = (32 + lane_id_2737)) {
            x2289[(lane_id_2737 + warp_id_2713)] = x2302[0];
          }

          __syncwarp();
        }

      }

      __syncthreads();
      /* mapWarp */
      for (int warp_id_2721 = (threadIdx.x / 32);(warp_id_2721 < 1);warp_id_2721 = (warp_id_2721 + (blockDim.x / 32))) {
        {
          float x2100[1];
          {
            float x2120[1];
            {
              float x2140[1];
              {
                float x2160[1];
                {
                  float x2180[1];
                  {
                    float x2200[1];
                    /* mapLane */
                    /* iteration count is exactly 1, no loop emitted */
                    int lane_id_2798 = (threadIdx.x % 32);
                    x2200[0] = ((((lane_id_2798 + (32 * warp_id_2721)) < 2)) ? (x2289[(lane_id_2798 + (32 * warp_id_2721))]) : (0.0f));
                    __syncwarp();
                    /* mapLane */
                    /* iteration count is exactly 1, no loop emitted */
                    int lane_id_2799 = (threadIdx.x % 32);
                    x2180[0] = (x2200[0] + __shfl_down_sync(0xFFFFFFFF, x2200[0], 16));
                    __syncwarp();
                  }

                  /* mapLane */
                  /* iteration count is exactly 1, no loop emitted */
                  int lane_id_2797 = (threadIdx.x % 32);
                  x2160[0] = (x2180[0] + __shfl_down_sync(0xFFFFFFFF, x2180[0], 8));
                  __syncwarp();
                }

                /* mapLane */
                /* iteration count is exactly 1, no loop emitted */
                int lane_id_2794 = (threadIdx.x % 32);
                x2140[0] = (x2160[0] + __shfl_down_sync(0xFFFFFFFF, x2160[0], 4));
                __syncwarp();
              }

              /* mapLane */
              /* iteration count is exactly 1, no loop emitted */
              int lane_id_2790 = (threadIdx.x % 32);
              x2120[0] = (x2140[0] + __shfl_down_sync(0xFFFFFFFF, x2140[0], 2));
              __syncwarp();
            }

            /* mapLane */
            /* iteration count is exactly 1, no loop emitted */
            int lane_id_2785 = (threadIdx.x % 32);
            x2100[0] = (x2120[0] + __shfl_down_sync(0xFFFFFFFF, x2120[0], 1));
            __syncwarp();
          }

          /* mapLane */
          for (int lane_id_2779 = (threadIdx.x % 32);(lane_id_2779 < 1);lane_id_2779 = (32 + lane_id_2779)) {
            output[((block_id_2687 + lane_id_2779) + warp_id_2721)] = x2100[0];
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
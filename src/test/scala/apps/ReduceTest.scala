package apps

import rise.cuda.DSL._
import rise.OpenCL.DSL.{oclReduceSeq, toLocal, toPrivate, toPrivateFun}
import rise.core.DSL._
import rise.core.HighLevelConstructs.reorderWithStride
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

  test("block reduce w/o reduceSeq"){
    val deviceTest = {

      val op = fun(f32 x f32)(t => t._1 + t._2)

      // reduceDevice: (n: nat) -> n.f32 -> n/numElemsBlock.f32, where n % numElemsBlock = 0
      nFun(n =>
        fun(n `.` f32)(blockChunk =>
          blockChunk |> split(warpSize) |>
            mapWarp('x')(fun(warpChunk =>
              warpChunk |>
                mapLane('x')(id) |>
                toPrivate |>
                warpReduce(op)
            )) |>
            toLocal |>
            join |>
            padCst(0)(warpSize-(n /^ warpSize))(l(0f)) |>
            split(warpSize) |>
            mapWarp(warpReduce(op)) |>
            join
        )
      )
    }
    //64 blocks, 32 warps each; 32 elems per lane
    gen.cuKernel(deviceTest(1024), "blockReduceGenerated")
  }

  test("device reduce test3"){
    val deviceTest = {

      val op = fun(f32 x f32)(t => t._1 + t._2)

      // reduceDevice: (n: nat) -> n.f32 -> n/numElemsBlock.f32, where n % numElemsBlock = 0
      nFun((n, numElemsBlock, numElemsWarp) =>
        fun(n `.` f32)(arr =>
          arr |> split(numElemsBlock) |> // n/numElemsBlock.numElemsBlock.f32
            mapBlock('x')(fun(blockChunk =>
              blockChunk |> split(numElemsWarp /^ warpSize) |>
                mapThreads('x')(fun(threadChunk =>
                threadChunk |>
                  oclReduceSeq(AddressSpace.Private)(add)(l(0f))
                  )) |>
                toPrivate |>
                split(warpSize) |>
                mapWarp(warpReduce(op)) |>
                toLocal |> //(k/j).1.f32 where (k/j) = #warps per block
                join |> //(k/j).f32 where (k/j) = #warps per block
                padCst(0)(warpSize-(numElemsBlock /^ numElemsWarp))(l(0f)) |> //32.f32
                split(warpSize) |> //1.32.f32 in order to execute following reduction with one warp
                mapWarp(warpReduce(op)) |>
                join
            ))
        ))
    }
    //64 blocks, 32 warps each; 32 elems per lane
    gen.cuKernel(deviceTest(2097152)(32768)(1024), "deviceReduceGenerated")
  }

  test("device reduce test"){
    val deviceTest = {

      val op = fun(f32 x f32)(t => t._1 + t._2)

      // reduceDevice: (n: nat) -> n.f32 -> n/numElemsBlock.f32, where n % numElemsBlock = 0
      nFun((n, numElemsBlock, numElemsWarp) =>
        fun(n `.` f32)(arr =>
          arr |> reorderWithStride(128) |>
            split(numElemsBlock) |> // n/numElemsBlock.numElemsBlock.f32
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
    gen.cuKernel(deviceTest(2097152)(32768)(1024), "deviceReduceGenerated")
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

  //Generierter Code mit Coalescing
  /*
  extern "C" __global__
void deviceReduceGenerated(float* __restrict__ output, const float* __restrict__ x0){
  extern __shared__ char dynamicSharedMemory1521[];
  /* mapBlock */
  for (int block_id_1408 = blockIdx.x;(block_id_1408 < 64);block_id_1408 = (block_id_1408 + gridDim.x)) {
    {
      float* x991 = ((float*)(&(dynamicSharedMemory1521[0])));
      /* mapWarp */
      for (int warp_id_1434 = (threadIdx.x / 32);(warp_id_1434 < 32);warp_id_1434 = (warp_id_1434 + (blockDim.x / 32))) {
        {
          float x1004[1];
          {
            float x1024[1];
            {
              float x1044[1];
              {
                float x1064[1];
                {
                  float x1084[1];
                  {
                    float x1104[1];
                    {
                      float x1112[1];
                      /* mapLane */
                      /* iteration count is exactly 1, no loop emitted */
                      int lane_id_1484 = (threadIdx.x % 32);
                      /* oclReduceSeq */
                      {
                        float x1124;
                        x1124 = 0.0f;
                        for (int i_1486 = 0;(i_1486 < 32);i_1486 = (1 + i_1486)) {
                          x1124 = (x1124 + x0[(((((i_1486 + (32 * lane_id_1484)) + (1024 * warp_id_1434)) / 16384) + (2 * block_id_1408)) + (128 * (((i_1486 + (32 * lane_id_1484)) + (1024 * warp_id_1434)) % 16384)))]);
                        }

                        x1112[0] = x1124;
                      }

                      __syncwarp();
                      /* mapLane */
                      /* iteration count is exactly 1, no loop emitted */
                      int lane_id_1485 = (threadIdx.x % 32);
                      x1104[0] = x1112[0];
                      __syncwarp();
                    }

                    /* mapLane */
                    /* iteration count is exactly 1, no loop emitted */
                    int lane_id_1483 = (threadIdx.x % 32);
                    x1084[0] = (x1104[0] + __shfl_down_sync(0xFFFFFFFF, x1104[0], 16));
                    __syncwarp();
                  }

                  /* mapLane */
                  /* iteration count is exactly 1, no loop emitted */
                  int lane_id_1480 = (threadIdx.x % 32);
                  x1064[0] = (x1084[0] + __shfl_down_sync(0xFFFFFFFF, x1084[0], 8));
                  __syncwarp();
                }

                /* mapLane */
                /* iteration count is exactly 1, no loop emitted */
                int lane_id_1476 = (threadIdx.x % 32);
                x1044[0] = (x1064[0] + __shfl_down_sync(0xFFFFFFFF, x1064[0], 4));
                __syncwarp();
              }

              /* mapLane */
              /* iteration count is exactly 1, no loop emitted */
              int lane_id_1471 = (threadIdx.x % 32);
              x1024[0] = (x1044[0] + __shfl_down_sync(0xFFFFFFFF, x1044[0], 2));
              __syncwarp();
            }

            /* mapLane */
            /* iteration count is exactly 1, no loop emitted */
            int lane_id_1465 = (threadIdx.x % 32);
            x1004[0] = (x1024[0] + __shfl_down_sync(0xFFFFFFFF, x1024[0], 1));
            __syncwarp();
          }

          /* mapLane */
          for (int lane_id_1458 = (threadIdx.x % 32);(lane_id_1458 < 1);lane_id_1458 = (32 + lane_id_1458)) {
            x991[(lane_id_1458 + warp_id_1434)] = x1004[0];
          }

          __syncwarp();
        }

      }

      __syncthreads();
      /* mapWarp */
      for (int warp_id_1442 = (threadIdx.x / 32);(warp_id_1442 < 1);warp_id_1442 = (warp_id_1442 + (blockDim.x / 32))) {
        {
          float x802[1];
          {
            float x822[1];
            {
              float x842[1];
              {
                float x862[1];
                {
                  float x882[1];
                  {
                    float x902[1];
                    /* mapLane */
                    /* iteration count is exactly 1, no loop emitted */
                    int lane_id_1519 = (threadIdx.x % 32);
                    x902[0] = x991[(lane_id_1519 + (32 * warp_id_1442))];
                    __syncwarp();
                    /* mapLane */
                    /* iteration count is exactly 1, no loop emitted */
                    int lane_id_1520 = (threadIdx.x % 32);
                    x882[0] = (x902[0] + __shfl_down_sync(0xFFFFFFFF, x902[0], 16));
                    __syncwarp();
                  }

                  /* mapLane */
                  /* iteration count is exactly 1, no loop emitted */
                  int lane_id_1518 = (threadIdx.x % 32);
                  x862[0] = (x882[0] + __shfl_down_sync(0xFFFFFFFF, x882[0], 8));
                  __syncwarp();
                }

                /* mapLane */
                /* iteration count is exactly 1, no loop emitted */
                int lane_id_1515 = (threadIdx.x % 32);
                x842[0] = (x862[0] + __shfl_down_sync(0xFFFFFFFF, x862[0], 4));
                __syncwarp();
              }

              /* mapLane */
              /* iteration count is exactly 1, no loop emitted */
              int lane_id_1511 = (threadIdx.x % 32);
              x822[0] = (x842[0] + __shfl_down_sync(0xFFFFFFFF, x842[0], 2));
              __syncwarp();
            }

            /* mapLane */
            /* iteration count is exactly 1, no loop emitted */
            int lane_id_1506 = (threadIdx.x % 32);
            x802[0] = (x822[0] + __shfl_down_sync(0xFFFFFFFF, x822[0], 1));
            __syncwarp();
          }

          /* mapLane */
          for (int lane_id_1500 = (threadIdx.x % 32);(lane_id_1500 < 1);lane_id_1500 = (32 + lane_id_1500)) {
            output[((block_id_1408 + lane_id_1500) + warp_id_1442)] = x802[0];
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

}

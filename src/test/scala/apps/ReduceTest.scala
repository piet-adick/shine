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
  //val srcLanes = generate(fun(IndexType(warpSize))(i => i))

  private val id = fun(x => x)
  private def warpReduce(op: Expr): Expr = {
    fun(warpChunk =>
      warpChunk |>
        toPrivateFun(mapLane(id)) |> //32.f32
        let(fun(x => zip(x, x |> shflDownWarp(16)))) |> //32.(f32 x f32)
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

      // reduceDevice: (n: nat) -> n.f32 -> n/numElemsBlock.f32, where n % numElemsBlock = 0
      nFun(n =>
        nFun(numElemsBlock =>
          nFun(numElemsWarp => fun(n `.` f32)(arr =>
            arr |> split(numElemsBlock) |> // n/numElemsBlock.numElemsBlock.f32
              mapBlock('x')(fun(chunk =>
                chunk |> split(numElemsWarp) |> // numElemsBlock/numElemsWarp.numElemsWarp.f32
                  mapWarp('x')(fun(warpChunk =>
                    warpChunk |> split(numElemsWarp/warpSize) |> // warpSize.numElemsWarp/warpSize.f32
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
                  padCst(0)(warpSize-(numElemsBlock/numElemsWarp))(l(0f)) |> //32.f32
                  split(warpSize) |> //1.32.f32 in order to execute following reduction with one warp
                  mapWarp(warpReduce(op)) |>
                  join
              ))
          ))
        ))

    }
    gen.cuKernel(deviceTest(2048)(1024)(64), "deviceReduceTest")
  }

  // Generierter Code:
  /*
extern "C" __global__
void deviceReduceTest(float* __restrict__ output, int n0, int n1, int n2, const float* __restrict__ x0, __shared__ float* __restrict__ x842){
  /* Start of moved local vars */
  /* End of moved local vars */
  /* mapBlock */
  for (int block_id_1230 = blockIdx.x;(block_id_1230 < (n0 / n1));block_id_1230 = (block_id_1230 + gridDim.x)) {
    /* mapWarp */
    for (int warp_id_1239 = (threadIdx.x / 32);(warp_id_1239 < (n1 / n2));warp_id_1239 = (warp_id_1239 + (blockDim.x / 32))) {
      {
        float x855[(((31+(n2*(1/^(((n2) / (32))))))) / (32))];
        {
          float x873[(((31+(n2*(1/^(((n2) / (32))))))) / (32))];
          {
            float x891[(((31+(n2*(1/^(((n2) / (32))))))) / (32))];
            {
              float x909[(((31+(n2*(1/^(((n2) / (32))))))) / (32))];
              {
                float x927[(((31+(n2*(1/^(((n2) / (32))))))) / (32))];
                {
                  float x945[(((31+(n2*(1/^(((n2) / (32))))))) / (32))];
                  {
                    float x953[(((31+(n2*(1/^(((n2) / (32))))))) / (32))];
                    /* mapLane */
                    for (int lane_id_1289 = (threadIdx.x % 32);(lane_id_1289 < (n2 / (n2 / 32)));lane_id_1289 = (32 + lane_id_1289)) {
                      /* oclReduceSeq */
                      {
                        float x965;
                        x965 = 0.0f;
                        for (int i_1291 = 0;(i_1291 < (n2 / 32));i_1291 = (1 + i_1291)) {
                          x965 = (x965 + x0[(((i_1291 + (block_id_1230 * n1)) + (lane_id_1289 * (n2 / 32))) + (n2 * warp_id_1239))]);
                        }

                        x953[(lane_id_1289 / 32)] = x965;
                      }

                    }

                    /* mapLane */
                    for (int lane_id_1290 = (threadIdx.x % 32);(lane_id_1290 < (n2 / (n2 / 32)));lane_id_1290 = (32 + lane_id_1290)) {
                      x945[(lane_id_1290 / 32)] = x953[(lane_id_1290 / 32)];
                    }

                  }

                  /* mapLane */
                  for (int lane_id_1288 = (threadIdx.x % 32);(lane_id_1288 < (n2 / (n2 / 32)));lane_id_1288 = (32 + lane_id_1288)) {
                    x927[(lane_id_1288 / 32)] = (x945[(lane_id_1288 / 32)] + x945[(lane_id_1288 / 32)]);
                  }

                }

                /* mapLane */
                for (int lane_id_1285 = (threadIdx.x % 32);(lane_id_1285 < (n2 / (n2 / 32)));lane_id_1285 = (32 + lane_id_1285)) {
                  x909[(lane_id_1285 / 32)] = (x927[(lane_id_1285 / 32)] + x927[(lane_id_1285 / 32)]);
                }

              }

              /* mapLane */
              for (int lane_id_1281 = (threadIdx.x % 32);(lane_id_1281 < (n2 / (n2 / 32)));lane_id_1281 = (32 + lane_id_1281)) {
                x891[(lane_id_1281 / 32)] = (x909[(lane_id_1281 / 32)] + x909[(lane_id_1281 / 32)]);
              }

            }

            /* mapLane */
            for (int lane_id_1276 = (threadIdx.x % 32);(lane_id_1276 < (n2 / (n2 / 32)));lane_id_1276 = (32 + lane_id_1276)) {
              x873[(lane_id_1276 / 32)] = (x891[(lane_id_1276 / 32)] + x891[(lane_id_1276 / 32)]);
            }

          }

          /* mapLane */
          for (int lane_id_1270 = (threadIdx.x % 32);(lane_id_1270 < (n2 / (n2 / 32)));lane_id_1270 = (32 + lane_id_1270)) {
            x855[(lane_id_1270 / 32)] = (x873[(lane_id_1270 / 32)] + x873[(lane_id_1270 / 32)]);
          }

        }

        /* mapLane */
        for (int lane_id_1263 = (threadIdx.x % 32);(lane_id_1263 < 1);lane_id_1263 = (32 + lane_id_1263)) {
          x842[(lane_id_1263 + warp_id_1239)] = x855[0];
        }

      }

    }

    /* mapWarp */
    for (int warp_id_1247 = (threadIdx.x / 32);(warp_id_1247 < ((1 + ((-1 * (n1 / n2)) / 32)) + (n1 / (n2 * 32))));warp_id_1247 = (warp_id_1247 + (blockDim.x / 32))) {
      {
        float x663[1];
        {
          float x681[1];
          {
            float x699[1];
            {
              float x717[1];
              {
                float x735[1];
                {
                  float x753[1];
                  /* mapLane */
                  /* iteration count is exactly 1, no loop emitted */
                  int lane_id_1324 = (threadIdx.x % 32);
                  x753[0] = ((((lane_id_1324 + (32 * warp_id_1247)) < (n1 / n2))) ? (x842[(lane_id_1324 + (32 * warp_id_1247))]) : (0.0f));
                  /* mapLane */
                  /* iteration count is exactly 1, no loop emitted */
                  int lane_id_1325 = (threadIdx.x % 32);
                  x735[0] = (x753[0] + x753[0]);
                }

                /* mapLane */
                /* iteration count is exactly 1, no loop emitted */
                int lane_id_1323 = (threadIdx.x % 32);
                x717[0] = (x735[0] + x735[0]);
              }

              /* mapLane */
              /* iteration count is exactly 1, no loop emitted */
              int lane_id_1320 = (threadIdx.x % 32);
              x699[0] = (x717[0] + x717[0]);
            }

            /* mapLane */
            /* iteration count is exactly 1, no loop emitted */
            int lane_id_1316 = (threadIdx.x % 32);
            x681[0] = (x699[0] + x699[0]);
          }

          /* mapLane */
          /* iteration count is exactly 1, no loop emitted */
          int lane_id_1311 = (threadIdx.x % 32);
          x663[0] = (x681[0] + x681[0]);
        }

        /* mapLane */
        for (int lane_id_1305 = (threadIdx.x % 32);(lane_id_1305 < 1);lane_id_1305 = (32 + lane_id_1305)) {
          output[((((block_id_1230 + lane_id_1305) + warp_id_1247) + (((-1 * block_id_1230) * (n1 / n2)) / 32)) + ((block_id_1230 * n1) / (n2 * 32)))] = x663[0];
        }

      }

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
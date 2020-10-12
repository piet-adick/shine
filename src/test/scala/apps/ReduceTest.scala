package apps

import rise.cuda.DSL._
import rise.OpenCL.DSL.{oclReduceSeq, oclReduceSeqUnroll, toLocal, toPrivate, toPrivateFun}
import rise.core.DSL._
import rise.core._
import rise.core.TypeLevelDSL._
import rise.core.types._
import util.gen

class ReduceTest extends shine.test_util.Tests {

  val warpSize = 32
  private val id = fun(x => x)

  // 32.f32 -> 1.f32
  private def reduceWarpDown32(op: Expr): Expr = {
    fun(warpChunk =>
      warpChunk |>
        toPrivateFun(mapLane('x')(id)) |> //32.f32
        let(fun(x => zip(x, x |> shflDownWarp(16)))) |> //32.(f32 x f32)
        toPrivateFun(mapLane('x')(op)) |> //32.f32
        let(fun(x => zip(x, x |> shflDownWarp(8)))) |> //32.(f32 x f32)
        toPrivateFun(mapLane('x')(op)) |> //32.f32
        let(fun(x => zip(x, x |> shflDownWarp(4)))) |> //32.(f32 x f32)
        toPrivateFun(mapLane('x')(op)) |> //32.f32
        let(fun(x => zip(x, x |> shflDownWarp(2)))) |> //32.(f32 x f32)
        toPrivateFun(mapLane('x')(op)) |> //32.f32
        let(fun(x => zip(x, x |> shflDownWarp(1)))) |> //32.(f32 x f32)
        toPrivateFun(mapLane('x')(op)) |> //32.f32
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
        mapLane('x')(id) //1.f32
    )
  }
/*
  private def reduceWarpUp32(op: Expr): Expr = {
    fun(warpChunk =>
      warpChunk |>
        toPrivateFun(mapLane('x')(id)) |> //32.f32
        let(fun(x => zip(x, x |> shflUpWarp(16)))) |> //32.(f32 x f32)
        toPrivateFun(mapLane('x')(op)) |> //32.f32
        let(fun(x => zip(x, x |> shflUpWarp(8)))) |> //32.(f32 x f32)
        toPrivateFun(mapLane('x')(op)) |> //32.f32
        let(fun(x => zip(x, x |> shflUpWarp(4)))) |> //32.(f32 x f32)
        toPrivateFun(mapLane('x')(op)) |> //32.f32
        let(fun(x => zip(x, x |> shflUpWarp(2)))) |> //32.(f32 x f32)
        toPrivateFun(mapLane('x')(op)) |> //32.f32
        let(fun(x => zip(x, x |> shflUpWarp(1)))) |> //32.(f32 x f32)
        toPrivateFun(mapLane('x')(op)) |> //32.f32
        drop(31) |> //1.f32
        mapLane('x')(id) //1.f32
    )
  }
*/
  private def reduceWarpXor32(op: Expr): Expr = {
    fun(warpChunk =>
      warpChunk |>
        toPrivateFun(mapLane('x')(id)) |> //32.f32
        let(fun(x => zip(x, x |> shflXorWarp(1)))) |> //32.(f32 x f32)
        toPrivateFun(mapLane('x')(op)) |> //32.f32
        let(fun(x => zip(x, x |> shflXorWarp(2)))) |> //32.(f32 x f32)
        toPrivateFun(mapLane('x')(op)) |> //32.f32
        let(fun(x => zip(x, x |> shflXorWarp(4)))) |> //32.(f32 x f32)
        toPrivateFun(mapLane('x')(op)) |> //32.f32
        let(fun(x => zip(x, x |> shflXorWarp(8)))) |> //32.(f32 x f32)
        toPrivateFun(mapLane('x')(op)) |> //32.f32
        let(fun(x => zip(x, x |> shflXorWarp(16)))) |> //32.(f32 x f32)
        toPrivateFun(mapLane('x')(op)) |> //32.f32
        take(1) |> //1.f32
        mapLane('x')(id) //1.f32
    )
  }

  // n.f32 -> f32
  private def reduceBlock(op: Expr): Expr = {
    implN(n =>
      fun(n `.` f32)(blockChunk =>
        blockChunk |>
          padCst(0)((warpSize-(n%warpSize))%warpSize)(l(0f)) |> // pad to next multiple of warpSize
          split(warpSize) |>
          mapWarp('x')(fun(warpChunk =>
            warpChunk |>
              mapLane('x')(id) |>
              toPrivate |>
              reduceWarpDown32(op)
          )) |>
          toLocal |>
          join |>
          padCst(0)(warpSize-(n / warpSize))(l(0f)) |> // pad to warpSize
          split(warpSize) |>
          mapWarp('x')(reduceWarpDown32(op)) |>
          join
      )
    )
  }

  test("block reduce test"){

    val op = fun(f32 x f32)(t => t._1 + t._2)

    val blockTest = {
      nFun(n =>
        fun(n `.` f32)(arr =>
          arr |>
            reduceBlock(op)
        )
      )
    }

    gen.cuKernel(blockTest, "blockReduceGenerated")
  }

  test("device reduce test (grid stride)"){

    val op = fun(f32 x f32)(t => t._1 + t._2)

    val deviceTest = {
      nFun(n =>
        fun(n `.` f32)(arr =>
          arr |>
            split(shine.cuda.globalDim('x')) |>
            transpose |>
            mapGlobal('x')(oclReduceSeq(AddressSpace.Private)(add)(l(0f))) |>
            toPrivate |>
            // pad to next multiple of 1024 (ArithExpr doesn't simplify this properly)
            //padCst(0)((1024-(n%1024))%1024)(l(0f)) |>
            split(1024) |>
            mapBlock('x')(reduceBlock(op))
        )
      )
    }

    gen.cuKernel(deviceTest, "deviceReduceGenerated")
  }

  test("device reduce test (naive smem)"){
    val deviceTest = {
      // reduceDevice: (n: nat) -> n.f32 -> n/numElemsBlock.f32, where n % numElemsBlock = 0
      nFun((elemsBlock, elemsSeq, n) =>
        fun(n `.` f32)(arr =>
          arr |> padCst(0)((elemsBlock-(n%elemsBlock))%elemsBlock)(l(0f)) |>
            split(elemsBlock) |> // n/numElemsBlock.numElemsBlock.f32
            mapBlock('x')(fun(chunk =>
              chunk |> split(elemsSeq) |> // numElemsBlock/numElemsWarp.numElemsWarp.f32
                mapThreads('x')(fun(seqChunk =>
                  seqChunk |>
                    oclReduceSeqUnroll(AddressSpace.Private)(add)(l(0f))
                )) |> toLocal |>
              split(elemsBlock) |>
                mapThreads('x')(fun(seqChunk =>
                  seqChunk |>
                    oclReduceSeq(AddressSpace.Private)(add)(l(0f))
                ))
            ))
        ))
    }

    gen.cuKernel(deviceTest(16384)(16), "deviceReduceGenerated_smem_NoSeq")
  }

  test("device reduce test (fast smem)"){
    val op = fun(f32 x f32)(t => t._1 + t._2)

    val deviceTest = {
      // reduceDevice: (n: nat) -> n.f32 -> n/numElemsBlock.f32, where n % numElemsBlock = 0
      nFun((elemsBlock, n) =>
        fun(n `.` f32)(arr =>
          arr |> padCst(0)((elemsBlock-(n%elemsBlock))%elemsBlock)(l(0f)) |>
            split(elemsBlock) |> // n/numElemsBlock.numElemsBlock.f32
            mapBlock('x')(fun(chunk =>
              chunk |> split(elemsBlock/1024) |> // numElemsBlock/numElemsWarp.numElemsWarp.f32
                mapThreads('x')(fun(seqChunk =>
                  seqChunk |>
                    oclReduceSeqUnroll(AddressSpace.Private)(add)(l(0f))
                )) |> toLocal |>
                fun(x => zip(x |> take(512))(x |> drop(512))) |>
                mapThreads('x')(op) |> toLocal |>
                fun(x => zip(x |> take(256))(x |> drop(256))) |>
                mapThreads('x')(op) |> toLocal |>
                fun(x => zip(x |> take(128))(x |> drop(128))) |>
                mapThreads('x')(op) |> toLocal |>
                fun(x => zip(x |> take(64))(x |> drop(64))) |>
                mapThreads('x')(op) |> toLocal |>
                fun(x => zip(x |> take(32))(x |> drop(32))) |>
                mapThreads('x')(op) |> toLocal |>
                fun(x => zip(x |> take(16))(x |> drop(16))) |>
                mapThreads('x')(op) |> toLocal |>
                fun(x => zip(x |> take(8))(x |> drop(8))) |>
                mapThreads('x')(op) |> toLocal |>
                fun(x => zip(x |> take(4))(x |> drop(4))) |>
                mapThreads('x')(op) |> toLocal |>
                fun(x => zip(x |> take(2))(x |> drop(2))) |>
                mapThreads('x')(op) |> toLocal |>
                fun(x => zip(x |> take(1))(x |> drop(1))) |>
                mapThreads('x')(op)
            ))
        ))
    }

    gen.cuKernel(deviceTest(16384), "deviceReduceGenerated_smem_NoSeq")
  }

  test("device reduce test (fast smemghjghj)"){
    val deviceTest = {
      // reduceDevice: (n: nat) -> n.f32 -> n/numElemsBlock.f32, where n % numElemsBlock = 0
      nFun((elemsBlock, n) =>
        fun(n `.` f32)(arr =>
          arr |> padCst(0)((elemsBlock-(n%elemsBlock))%elemsBlock)(l(0f)) |>
            split(elemsBlock) |> // n/numElemsBlock.numElemsBlock.f32
            mapBlock('x')(fun(chunk =>
              chunk |> split(elemsBlock/1024) |> // numElemsBlock/numElemsWarp.numElemsWarp.f32
                mapThreads('x')(fun(seqChunk =>
                  seqChunk |>
                    oclReduceSeqUnroll(AddressSpace.Private)(add)(l(0f))
                )) |> toLocal |>
                split(2) |>
                mapThreads('x')(fun(seqChunk =>
                  seqChunk |>
                    oclReduceSeqUnroll(AddressSpace.Local)(add)(l(0f))
                )) |>
                split(2) |>
                mapThreads('x')(fun(seqChunk =>
                  seqChunk |>
                    oclReduceSeqUnroll(AddressSpace.Local)(add)(l(0f))
                )) |>
                split(2) |>
                mapThreads('x')(fun(seqChunk =>
                  seqChunk |>
                    oclReduceSeqUnroll(AddressSpace.Local)(add)(l(0f))
                )) |>
                split(2) |>
                mapThreads('x')(fun(seqChunk =>
                  seqChunk |>
                    oclReduceSeqUnroll(AddressSpace.Local)(add)(l(0f))
                )) |>
                split(2) |>
                mapThreads('x')(fun(seqChunk =>
                  seqChunk |>
                    oclReduceSeqUnroll(AddressSpace.Local)(add)(l(0f))
                )) |>
                split(2) |>
                mapThreads('x')(fun(seqChunk =>
                  seqChunk |>
                    oclReduceSeqUnroll(AddressSpace.Local)(add)(l(0f))
                )) |>
                split(2) |>
                mapThreads('x')(fun(seqChunk =>
                  seqChunk |>
                    oclReduceSeqUnroll(AddressSpace.Local)(add)(l(0f))
                )) |>
                split(2) |>
                mapThreads('x')(fun(seqChunk =>
                  seqChunk |>
                    oclReduceSeqUnroll(AddressSpace.Local)(add)(l(0f))
                )) |>
                split(2) |>
                mapThreads('x')(fun(seqChunk =>
                  seqChunk |>
                    oclReduceSeqUnroll(AddressSpace.Private)(add)(l(0f))
                ))
            ))
        ))
    }

    gen.cuKernel(deviceTest(16384), "deviceReduceGenerated_smem_NoSeq")
  }

  test("device reduce test (shfl xor)"){
    val deviceTest = {

      val op = fun(f32 x f32)(t => t._1 + t._2)

      // reduceDevice: (n: nat) -> n.f32 -> n/numElemsBlock.f32, where n % numElemsBlock = 0
      nFun((elemsBlock, elemsWarp, n) =>
        fun(n `.` f32)(arr =>
          arr |> padCst(0)((elemsBlock-(n%elemsBlock))%elemsBlock)(l(0f)) |>
            split(elemsBlock) |> // n/numElemsBlock.numElemsBlock.f32
            mapBlock('x')(fun(chunk =>
              chunk |> split(elemsWarp) |> // numElemsBlock/numElemsWarp.numElemsWarp.f32
                mapWarp('x')(fun(warpChunk =>
                  warpChunk |> split(warpSize) |> // warpSize.numElemsWarp/warpSize.f32
                    transpose |>
                    mapLane(fun(threadChunk =>
                      threadChunk |>
                        oclReduceSeqUnroll(AddressSpace.Private)(add)(l(0f))
                    )) |> toPrivate |>
                    reduceWarpXor32(op)
                )) |> toLocal |> //(k/j).1.f32 where (k/j) = #warps per block
                join |> //(k/j).f32 where (k/j) = #warps per block
                padCst(0)(warpSize-(elemsBlock /^ elemsWarp))(l(0f)) |> //32.f32
                split(warpSize) |> //1.32.f32 in order to execute following reduction with one warp
                mapWarp(reduceWarpXor32(op)) |>
                join
            ))
        ))
    }

    gen.cuKernel(deviceTest(1024)(32), "deviceReduceGenerated_Xor_NoSeq")

    gen.cuKernel(deviceTest(1024)(128), "deviceReduceGenerated_Xor_1024_128")
    gen.cuKernel(deviceTest(2048)(256), "deviceReduceGenerated_Xor_2048_256")
    gen.cuKernel(deviceTest(4096)(512), "deviceReduceGenerated_Xor_4096_512")

    gen.cuKernel(deviceTest(2048)(128), "deviceReduceGenerated_Xor_2048_128")
    gen.cuKernel(deviceTest(4096)(256), "deviceReduceGenerated_Xor_4096_256")
    gen.cuKernel(deviceTest(8192)(512), "deviceReduceGenerated_Xor_8192_512")

    gen.cuKernel(deviceTest(4096)(128), "deviceReduceGenerated_Xor_4096_128")
    gen.cuKernel(deviceTest(8192)(256), "deviceReduceGenerated_Xor_8192_256")
    gen.cuKernel(deviceTest(16384)(512), "deviceReduceGenerated_Xor_16384_512")
  }

  test("device reduce test (shfl down)"){
    val deviceTest = {

      val op = fun(f32 x f32)(t => t._1 + t._2)

      // reduceDevice: (n: nat) -> n.f32 -> n/numElemsBlock.f32, where n % numElemsBlock = 0
      nFun((elemsBlock, elemsWarp, n) =>
        fun(n `.` f32)(arr =>
          arr |> padCst(0)((elemsBlock-(n%elemsBlock))%elemsBlock)(l(0f)) |>
            split(elemsBlock) |> // n/numElemsBlock.numElemsBlock.f32
            mapBlock('x')(fun(chunk =>
              chunk |> split(elemsWarp) |> // numElemsBlock/numElemsWarp.numElemsWarp.f32
                mapWarp('x')(fun(warpChunk =>
                  warpChunk |> split(warpSize) |> // warpSize.numElemsWarp/warpSize.f32
                    transpose |>
                    mapLane(fun(threadChunk =>
                      threadChunk |>
                        oclReduceSeqUnroll(AddressSpace.Private)(add)(l(0f))
                    )) |> toPrivate |>
                    reduceWarpDown32(op)
                )) |> toLocal |> //(k/j).1.f32 where (k/j) = #warps per block
                join |> //(k/j).f32 where (k/j) = #warps per block
                padCst(0)(warpSize-(elemsBlock /^ elemsWarp))(l(0f)) |> //32.f32
                split(warpSize) |> //1.32.f32 in order to execute following reduction with one warp
                mapWarp(reduceWarpDown32(op)) |>
                join
            ))
        ))
    }

    gen.cuKernel(deviceTest(1024)(32), "deviceReduceGenerated_Down_NoSeq")

    gen.cuKernel(deviceTest(1024)(128), "deviceReduceGenerated_Down_1024_128")
    gen.cuKernel(deviceTest(2048)(256), "deviceReduceGenerated_Down_2048_256")
    gen.cuKernel(deviceTest(4096)(512), "deviceReduceGenerated_Down_4096_512")

    gen.cuKernel(deviceTest(2048)(128), "deviceReduceGenerated_Down_2048_128")
    gen.cuKernel(deviceTest(4096)(256), "deviceReduceGenerated_Down_4096_256")
    gen.cuKernel(deviceTest(8192)(512), "deviceReduceGenerated_Down_8192_512")

    gen.cuKernel(deviceTest(4096)(128), "deviceReduceGenerated_Down_4096_128")
    gen.cuKernel(deviceTest(8192)(256), "deviceReduceGenerated_Down_8192_256")
    gen.cuKernel(deviceTest(16384)(512), "deviceReduceGenerated_Down_16384_512")
  }

}

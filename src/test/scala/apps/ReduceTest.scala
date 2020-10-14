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
        take(1) |> //1.f32
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



  test("device reduce test (naive smem)"){
    val deviceTest = {
      // reduceDevice: (n: nat) -> n.f32 -> n/numElemsBlock.f32, where n % numElemsBlock = 0
      nFun((elemsBlock, n) =>
        fun(n `.` f32)(arr =>
          arr |> padCst(0)((elemsBlock-(n%elemsBlock))%elemsBlock)(l(0f)) |>
            split(elemsBlock) |> // n/numElemsBlock.numElemsBlock.f32
            mapBlock('x')(fun(chunk =>
              chunk |> split(elemsBlock/^1024) |> // numElemsBlock/numElemsWarp.numElemsWarp.f32
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

    gen.cuKernel(deviceTest(1024), "deviceReduceGeneratedSharedNaive_1024")
    gen.cuKernel(deviceTest(2048), "deviceReduceGeneratedSharedNaive_2048")
    gen.cuKernel(deviceTest(4096), "deviceReduceGeneratedSharedNaive_4096")
    gen.cuKernel(deviceTest(8192), "deviceReduceGeneratedSharedNaive_8192")
    gen.cuKernel(deviceTest(16384), "deviceReduceGeneratedSharedNaive_16384")  }

  test("device reduce test (fast smem)"){
    val op = fun(2`.`f32)(t => t`@`lidx(0, 2) + t`@`lidx(1, 2))

    val deviceTest = {
      // reduceDevice: (n: nat) -> n.f32 -> n/elemsBlock.f32 where n % numElemsBlock = 0, (n/elemsBlock)+1.f32 else
      nFun((elemsBlock, n) =>
        fun(n `.` f32)(arr =>
          arr |> padCst(0)((elemsBlock-(n%elemsBlock))%elemsBlock)(l(0f)) |>
            split(elemsBlock) |>
            mapBlock('x')(fun(chunk =>
              chunk |> split(elemsBlock/^1024) |>
                mapThreads('x')(fun(seqChunk =>
                  seqChunk |>
                    oclReduceSeqUnroll(AddressSpace.Private)(add)(l(0f))
                )) |> toLocal |>
                split(512) |>
                transpose |>
                mapThreads('x')(op) |>
                toLocal |>
                split(256) |>
                transpose |>
                mapThreads('x')(op) |>
                toLocal |>
                split(128) |>
                transpose |>
                mapThreads('x')(op) |>
                toLocal |>
                split(64) |>
                transpose |>
                mapThreads('x')(op) |>
                toLocal |>
                split(32) |>
                transpose |>
                mapThreads('x')(op) |>
                toLocal |>
                split(16) |>
                transpose |>
                mapThreads('x')(op) |>
                toLocal |>
                split(8) |>
                transpose |>
                mapThreads('x')(op) |>
                toLocal |>
                split(4) |>
                transpose |>
                mapThreads('x')(op) |>
                toLocal |>
                split(2) |>
                transpose |>
                mapThreads('x')(op) |>
                toLocal |>
                split(1) |>
                transpose |>
                mapThreads('x')(op)
            ))
        ))
    }

    gen.cuKernel(deviceTest(1024), "deviceReduceGeneratedShared_1024")
    gen.cuKernel(deviceTest(2048), "deviceReduceGeneratedShared_2048")
    gen.cuKernel(deviceTest(4096), "deviceReduceGeneratedShared_4096")
    gen.cuKernel(deviceTest(8192), "deviceReduceGeneratedShared_8192")
    gen.cuKernel(deviceTest(16384), "deviceReduceGeneratedShared_16384")
  }

  test("device reduce test (shfl xor)"){
    val deviceTest = {

      val op = fun(f32 x f32)(t => t._1 + t._2)

      // reduceDevice: (n: nat) -> n.f32 -> n/elemsBlock.f32 where n % numElemsBlock = 0, (n/elemsBlock)+1.f32 else
      nFun((elemsBlock, elemsWarp, n) =>
        fun(n `.` f32)(arr =>
          arr |> padCst(0)((elemsBlock-(n%elemsBlock))%elemsBlock)(l(0f)) |>
            split(elemsBlock) |>
            mapBlock('x')(fun(chunk => // numElemsBlock.f32
              chunk |> split(elemsWarp) |>
                mapWarp('x')(fun(warpChunk => // numElemsWarp.f32
                  warpChunk |> split(warpSize) |>
                    transpose |>
                    mapLane(fun(threadChunk => // elemsSeq.f32
                      threadChunk |>
                        oclReduceSeqUnroll(AddressSpace.Private)(add)(l(0f))
                    )) |> toPrivate |>
                    reduceWarpXor32(op)
                )) |> toLocal |>
                join |> // (#warps per block).f32
                padCst(0)(warpSize-(elemsBlock /^ elemsWarp))(l(0f)) |> //32.f32
                split(warpSize) |> //1.32.f32 in order to execute following reduction with one warp
                mapWarp(reduceWarpXor32(op)) |>
                join
            ))
        ))
    }

    // elemsBlock = 1-faches von elemsWarp
    gen.cuKernel(deviceTest(32)(32), "deviceReduceGenerated_ShflXor_32_32")
    gen.cuKernel(deviceTest(64)(64), "deviceReduceGenerated_ShflXor_64_64")
    gen.cuKernel(deviceTest(128)(128), "deviceReduceGenerated_ShflXor_128_128")
    gen.cuKernel(deviceTest(256)(256), "deviceReduceGenerated_ShflXor_256_256")
    gen.cuKernel(deviceTest(512)(512), "deviceReduceGenerated_ShflXor_512_512")

    // elemsBlock = 2-faches von elemsWarp
    gen.cuKernel(deviceTest(64)(32), "deviceReduceGenerated_ShflXor_64_32")
    gen.cuKernel(deviceTest(128)(64), "deviceReduceGenerated_ShflXor_128_64")
    gen.cuKernel(deviceTest(256)(128), "deviceReduceGenerated_ShflXor_256_128")
    gen.cuKernel(deviceTest(512)(256), "deviceReduceGenerated_ShflXor_512_256")
    gen.cuKernel(deviceTest(1024)(512), "deviceReduceGenerated_ShflXor_1024_512")

    // elemsBlock = 4-faches von elemsWarp
    gen.cuKernel(deviceTest(128)(32), "deviceReduceGenerated_ShflXor_128_32")
    gen.cuKernel(deviceTest(256)(64), "deviceReduceGenerated_ShflXor_256_64")
    gen.cuKernel(deviceTest(512)(128), "deviceReduceGenerated_ShflXor_512_128")
    gen.cuKernel(deviceTest(1024)(256), "deviceReduceGenerated_ShflXor_1024_256")
    gen.cuKernel(deviceTest(2048)(512), "deviceReduceGenerated_ShflXor_2048_512")

    // elemsBlock = 8-faches von elemsWarp
    gen.cuKernel(deviceTest(256)(32), "deviceReduceGenerated_ShflXor_256_32")
    gen.cuKernel(deviceTest(512)(64), "deviceReduceGenerated_ShflXor_512_64")
    gen.cuKernel(deviceTest(1024)(128), "deviceReduceGenerated_ShflXor_1024_128")
    gen.cuKernel(deviceTest(2048)(256), "deviceReduceGenerated_ShflXor_2048_256")
    gen.cuKernel(deviceTest(4096)(512), "deviceReduceGenerated_ShflXor_4096_512")

    // elemsBlock = 16-faches von elemsWarp
    gen.cuKernel(deviceTest(512)(32), "deviceReduceGenerated_ShflXor_512_32")
    gen.cuKernel(deviceTest(1024)(64), "deviceReduceGenerated_ShflXor_1024_64")
    gen.cuKernel(deviceTest(2048)(128), "deviceReduceGenerated_ShflXor_2048_128")
    gen.cuKernel(deviceTest(4096)(256), "deviceReduceGenerated_ShflXor_4096_256")
    gen.cuKernel(deviceTest(8192)(512), "deviceReduceGenerated_ShflXor_8192_512")

    // elemsBlock = 32-faches von elemsWarp
    gen.cuKernel(deviceTest(1024)(32), "deviceReduceGenerated_ShflXor_1024_32")
    gen.cuKernel(deviceTest(2048)(64), "deviceReduceGenerated_ShflXor_2048_64")
    gen.cuKernel(deviceTest(4096)(128), "deviceReduceGenerated_ShflXor_4096_128")
    gen.cuKernel(deviceTest(8192)(256), "deviceReduceGenerated_ShflXor_8192_256")
    gen.cuKernel(deviceTest(16384)(512), "deviceReduceGenerated_ShflXor_16384_512")
  }

  test("device reduce test (shfl down)"){
    val deviceTest = {

      val op = fun(f32 x f32)(t => t._1 + t._2)

      // reduceDevice: (n: nat) -> n.f32 -> n/elemsBlock.f32 where n % numElemsBlock = 0, (n/elemsBlock)+1.f32 else
      nFun((elemsBlock, elemsWarp, n) =>
        fun(n `.` f32)(arr =>
          arr |> padCst(0)((elemsBlock-(n%elemsBlock))%elemsBlock)(l(0f)) |>
            split(elemsBlock) |>
            mapBlock('x')(fun(chunk => // numElemsBlock.f32
              chunk |> split(elemsWarp) |>
                mapWarp('x')(fun(warpChunk => // numElemsWarp.f32
                  warpChunk |> split(warpSize) |>
                    transpose |>
                    mapLane(fun(threadChunk => // elemsSeq.f32
                      threadChunk |>
                        oclReduceSeqUnroll(AddressSpace.Private)(add)(l(0f))
                    )) |> toPrivate |>
                    reduceWarpDown32(op)
                )) |> toLocal |>
                join |> // (#warps per block).f32
                padCst(0)(warpSize-(elemsBlock /^ elemsWarp))(l(0f)) |> //32.f32
                split(warpSize) |> //1.32.f32 in order to execute following reduction with one warp
                mapWarp(reduceWarpDown32(op)) |>
                join
            ))
        ))
    }

    // elemsBlock = 1-faches von elemsWarp
    gen.cuKernel(deviceTest(32)(32), "deviceReduceGenerated_ShflDown_32_32")
    gen.cuKernel(deviceTest(64)(64), "deviceReduceGenerated_ShflDown_64_64")
    gen.cuKernel(deviceTest(128)(128), "deviceReduceGenerated_ShflDown_128_128")
    gen.cuKernel(deviceTest(256)(256), "deviceReduceGenerated_ShflDown_256_256")
    gen.cuKernel(deviceTest(512)(512), "deviceReduceGenerated_ShflDown_512_512")

    // elemsBlock = 2-faches von elemsWarp
    gen.cuKernel(deviceTest(64)(32), "deviceReduceGenerated_ShflDown_64_32")
    gen.cuKernel(deviceTest(128)(64), "deviceReduceGenerated_ShflDown_128_64")
    gen.cuKernel(deviceTest(256)(128), "deviceReduceGenerated_ShflDown_256_128")
    gen.cuKernel(deviceTest(512)(256), "deviceReduceGenerated_ShflDown_512_256")
    gen.cuKernel(deviceTest(1024)(512), "deviceReduceGenerated_ShflDown_1024_512")

    // elemsBlock = 4-faches von elemsWarp
    gen.cuKernel(deviceTest(128)(32), "deviceReduceGenerated_ShflDown_128_32")
    gen.cuKernel(deviceTest(256)(64), "deviceReduceGenerated_ShflDown_256_64")
    gen.cuKernel(deviceTest(512)(128), "deviceReduceGenerated_ShflDown_512_128")
    gen.cuKernel(deviceTest(1024)(256), "deviceReduceGenerated_ShflDown_1024_256")
    gen.cuKernel(deviceTest(2048)(512), "deviceReduceGenerated_ShflDown_2048_512")

    // elemsBlock = 8-faches von elemsWarp
    gen.cuKernel(deviceTest(256)(32), "deviceReduceGenerated_ShflDown_256_32")
    gen.cuKernel(deviceTest(512)(64), "deviceReduceGenerated_ShflDown_512_64")
    gen.cuKernel(deviceTest(1024)(128), "deviceReduceGenerated_ShflDown_1024_128")
    gen.cuKernel(deviceTest(2048)(256), "deviceReduceGenerated_ShflDown_2048_256")
    gen.cuKernel(deviceTest(4096)(512), "deviceReduceGenerated_ShflDown_4096_512")

    // elemsBlock = 16-faches von elemsWarp
    gen.cuKernel(deviceTest(512)(32), "deviceReduceGenerated_ShflDown_512_32")
    gen.cuKernel(deviceTest(1024)(64), "deviceReduceGenerated_ShflDown_1024_64")
    gen.cuKernel(deviceTest(2048)(128), "deviceReduceGenerated_ShflDown_2048_128")
    gen.cuKernel(deviceTest(4096)(256), "deviceReduceGenerated_ShflDown_4096_256")
    gen.cuKernel(deviceTest(8192)(512), "deviceReduceGenerated_ShflDown_8192_512")

    // elemsBlock = 32-faches von elemsWarp
    gen.cuKernel(deviceTest(1024)(32), "deviceReduceGenerated_ShflDown_1024_32")
    gen.cuKernel(deviceTest(2048)(64), "deviceReduceGenerated_ShflDown_2048_64")
    gen.cuKernel(deviceTest(4096)(128), "deviceReduceGenerated_ShflDown_4096_128")
    gen.cuKernel(deviceTest(8192)(256), "deviceReduceGenerated_ShflDown_8192_256")
    gen.cuKernel(deviceTest(16384)(512), "deviceReduceGenerated_ShflDown_16384_512")
  }

}

/*
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
*/
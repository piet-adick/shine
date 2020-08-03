// WIP

package shine.cuda


import rise.OpenCL.TypedDSL.{oclReduceSeq, toGlobal, toPrivate}
import rise.core.TypedDSL.{FunPipe, add, fun, idx, l, padCst, split}
import rise.core.types.AddressSpace.Private
import rise.cuda.DSL.{mapBlock, mapThreads, mapWarp}
import util.gen


/*
import shine.DPIA.FunctionalPrimitives.Split
import shine.DPIA.Phrases.{BinOp, Identifier, Lambda, Literal, Operators}
import shine.DPIA.Semantics.OperationalSemantics.FloatData
import shine.DPIA.Types.{ArrayType, ExpType, f32, read}
import shine.OpenCL.FunctionalPrimitives.OpenCLReduceSeq
import shine.cuda.primitives.functional.{MapBlock, MapThreads, MapWarp}
 */

class DeviceReduceTest extends shine.test_util.Tests {

  test("deviceReduce") {

    val w = fun(arr =>
      arr |> idx(0)
    )

    val f = fun(arr =>
      arr |> split(2048)
        |> mapBlock(fun(chunk =>
        chunk |> split(4)
          |> mapThreads(fun(threadChunk =>
          threadChunk
            |> oclReduceSeq(Private)(add)
        )
        )
          |> split(32)
          |> mapWarp(fun(warpChunk =>
          warpChunk
            |> w //reduceWarp
        )
        )
          |> padCst(0)(16)(l(0f))
          |> split(32)
          |> mapWarp(fun(warpChunk =>
          warpChunk
            |> mapThreads(fun(threadValue =>
            threadValue
              |> toPrivate
          ))
            |> w //reduceWarp
        )
        )
      )
      )

      |> toGlobal
    )

    gen.cuKernel(f)

    /*
    val arrG = Identifier("arr", ExpType(ArrayType(4096, f32), read))
    val arrB = Identifier("arr", ExpType(ArrayType(2048, f32), read))
    val arrW = Identifier("arr", ExpType(ArrayType(32, f32), read))
    val arrT = Identifier("arr", ExpType(ArrayType(4, f32), read))
    val x = Identifier("x", ExpType(f32, read))
    val y = Identifier("y", ExpType(f32, read))
    val f =
      Lambda(arrG,
        MapBlock(0)(2, ArrayType(2048, f32), f32,
          Lambda(arrB,
            MapThreads(0)(512, ArrayType(4, f32), f32,
              Lambda(arrT,
                OpenCLReduceSeq(4, AddressSpace.Private, f32, f32, Lambda(x,Lambda(y,BinOp(Operators.Binary.ADD, x, y))), Literal(FloatData(0.0f)), arrT, false)
              ),
              Split(4, 512, read, f32, arrB)
            )
          ),
          Split(2048, 2, read, f32, arrG)
        )
      )

    val code = KernelGenerator().makeCode(f).code
    println(code)
    */

  }

}

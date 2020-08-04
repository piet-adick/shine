// WIP

package shine.cuda


import rise.OpenCL.TypedDSL._
import rise.core.TypeLevelDSL._
import rise.core.TypedDSL._
import rise.core.types.AddressSpace._
import rise.core.types._
import rise.cuda.DSL._
import util.gen


class DeviceReduceTest extends shine.test_util.Tests {

  test("deviceReduce") {

    val n = 8192
    val k = 2048
    val j = 4

    val reduceWarp = fun(arr =>
      arr |> idx(0)
    )

    val f = fun(n`.`f32)(arr =>
      arr
        |> split(k)
        |> mapBlock(fun(chunk =>
          chunk |> split(j)
          |> mapThreads(fun(threadChunk =>
            threadChunk
            |> oclReduceSeq(Private)(add)(l(0f))
          ))
          |> split(32)
          |> mapWarp(fun(warpChunk =>
            warpChunk
            |> printType("fst")
            |> reduceWarp
          ))
          |> padCst(0)(32-(k/j)/32)(l(0f))
          |> split(32)
          |> mapWarp(fun(warpChunk =>
            warpChunk
            |> mapThreads(fun(threadValue =>
            threadValue
              |> toPrivate
            ))
            |> printType("snd")
            |> reduceWarp
          ))
        ))
        |> toGlobal
      )

    gen.cuKernel(f)

  }

}

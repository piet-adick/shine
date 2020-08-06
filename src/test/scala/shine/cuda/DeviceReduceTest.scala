// WIP

package shine.cuda

import rise.cuda.DSL._
import rise.OpenCL.TypedDSL._
import rise.core.TypeLevelDSL._
import rise.core.TypedDSL._
import rise.core.types.AddressSpace._
import rise.core.types._
import util.gen


class DeviceReduceTest extends shine.test_util.Tests {

  test("deviceReduce") {

    val n = 8192
    val k = 2048
    val j = 4
    val redop = add

    val reduceWarp = fun(32`.`f32)(arr =>
      arr
      |> split(32)
      |> mapThreads(fun(threadChunk =>
        threadChunk
          |> oclReduceSeq(Private)(redop)(l(0f))
      ))
    )

    val f = fun(n`.`f32)(arr =>
      arr
        |> split(k)
        |> mapBlock(fun(chunk =>
          chunk |> split(j)
          |> mapThreads(fun(threadChunk =>
            threadChunk
            |> oclReduceSeq(Private)(redop)(l(0f))
          ))
          |> split(32)
          |> mapWarp(fun(warpChunk =>
            warpChunk
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
            |> reduceWarp
          ))
        ))
        |> toGlobal
      )

    gen.cuKernel(f)

  }

}

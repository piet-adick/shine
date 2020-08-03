package shine.OpenCL

import util.gen
import rise.core.DSL._
import rise.core.TypeLevelDSL._
import rise.core.types._
import rise.core.types.AddressSpace._
import rise.OpenCL.DSL._

class VariableThreadCount extends shine.test_util.Tests{

  test("test for varying amount of threads") {

    val e =
      fun(1024`.`f32)(x =>
        x |> split(512)
        |> mapWorkGroup(fun(chunk =>
          chunk
            |> split(32)
            |> mapLocal(fun(chunk =>
            chunk
              |> oclReduceSeq(Private)(fun(x => fun(a => a + x)))(l(0.0f))
            ))
            |> toLocal
            |> padCst(0)(16)(l(0.0f))
            |> split(32)
            |> mapLocal(fun(chunk =>
            chunk
              |> oclReduceSeq(Private)(fun(x => fun(a => a + x)))(l(0.0f))
          )
          )
        )
        )
      )

    gen.OpenCLKernel(LocalSize((16, 1, 1)), GlobalSize((32, 1, 1)))(e, "KERNEL")
  }
}

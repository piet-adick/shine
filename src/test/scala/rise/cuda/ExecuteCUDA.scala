package rise.OpenCL

import shine.OpenCL._
import rise.core._
import rise.core.DSL._
import rise.core.types._
import util.gen

class ExecuteCUDA extends shine.test_util.TestsWithExecutor {
  test("Running a simple kernel with generic input size") {
    val f: Expr = nFun(n => fun(ArrayType(n, int))(
      xs => xs |> mapSeq(fun(x => x + l(1)))))

    val kernel = gen.cuKernel(f)

    val kernelF = kernel.as[ScalaFunction `(` Int `,` Array[Int] `)=>` Array[Int]].withSizes(LocalSize(1), GlobalSize(1))
    val xs = Array.fill(8)(0)

    val (result, time) = kernelF(8 `,` xs)
    println(time)

    val gold = Array.fill(8)(1)
    assertResult(gold)(result)
  }
}

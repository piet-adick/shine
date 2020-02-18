import rise.core.DSL.{fun, mapSeq, nFun}
import rise.core.types.{ArrayType, VectorType, f32}
import util.gen

class Parameters extends shine.test_util.Tests {
  test("Output scalar") {
    gen.OpenCLKernel(fun(f32)(vs => vs))
  }

  test("Output scalar") {
    gen.cuKernel(fun(f32)(vs => vs))
  }
}
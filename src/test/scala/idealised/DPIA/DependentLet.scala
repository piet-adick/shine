package idealised.DPIA
import idealised.SurfaceLanguage.DSL._
import idealised.SurfaceLanguage.LiteralExpr
import idealised.SurfaceLanguage.Semantics.IntData
import idealised.SurfaceLanguage.Types._
import idealised.util.SyntaxChecker

class DependentLet extends idealised.util.Tests{

  test("Simple no capture") {
    val program = nFun(n =>
      fun(ArrayType(n, float))(xs =>
        letNat(fun(int)(x => x + 2), f =>
          letNat(fun(int)(x => x + 1), g =>
            letNat(5, five =>
              mapSeq(fun(x => x))(take(g(f(five())), xs))
            )
          )
        )
    ))

    val typed = TypeInference(program, Map())

    val p = idealised.OpenCL.KernelGenerator.makeCode(idealised.DPIA.FromSurfaceLanguage(typed))

    val code = p.code
    SyntaxChecker.checkOpenCL(code)
    println(code)
  }

  test("basic test") {
    val f = nFun(n => fun(IndexType(10))(idx =>
        fun(ArrayType(n, float))(xs =>
            letNat(idx,
              f => xs :>> take(f()) :>> mapSeq(fun(x => x + 1.0f)))
        )
      ))

    val typed = TypeInference(f, Map())

    val p = idealised.OpenCL.KernelGenerator.makeCode(idealised.DPIA.FromSurfaceLanguage(typed))

    val code = p.code
    SyntaxChecker.checkOpenCL(code)
    println(code)
  }
}

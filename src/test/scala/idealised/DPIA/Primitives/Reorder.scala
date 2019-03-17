package idealised.DPIA.Primitives

import idealised.OpenCL.SurfaceLanguage.DSL._
import idealised.SurfaceLanguage.DSL._
import idealised.SurfaceLanguage.Types._
import idealised.util.SyntaxChecker
import lift.arithmetic.Cst

class Reorder extends idealised.util.Tests {
  test("Simple gather example should generate syntactic valid C code with two one loops") {
    val e = nFun(n => fun(ArrayType(n, float))(xs =>
      xs :>> reorderWithStride(Cst(128)) :>> mapSeq(fun(x => x))
    ))

    val p = idealised.C.ProgramGenerator.makeCode(idealised.DPIA.FromSurfaceLanguage(TypeInference(e, Map())))
    val code = p.code
    SyntaxChecker(code)
    println(code)

    "for".r.findAllIn(code).length shouldBe 1
  }

  test("Simple 2D gather example should generate syntactic valid C code with two two loops") {
    val e = nFun(n => nFun(m => fun(ArrayType(n, ArrayType(m, float)))(xs =>
      xs :>> map(reorderWithStride(Cst(128))) :>> mapSeq(mapSeq(fun(x => x)))
    )))

    val p = idealised.C.ProgramGenerator.makeCode(idealised.DPIA.FromSurfaceLanguage(TypeInference(e, Map())))
    val code = p.code
    SyntaxChecker(code)
    println(code)

    "for".r.findAllIn(code).length shouldBe 2
  }

  test("Simple scatter example should generate syntactic valid C code with two one loops") {
    val e = nFun(n => fun(ArrayType(n, float))(xs =>
      xs :>> mapSeq(fun(x => x)) :>> reorderWithStride(Cst(128))
    ))

    val p = idealised.C.ProgramGenerator.makeCode(idealised.DPIA.FromSurfaceLanguage(TypeInference(e, Map())))
    val code = p.code
    SyntaxChecker(code)
    println(code)

    "for".r.findAllIn(code).length shouldBe 1
  }

  test("Simple 2D scatter example should generate syntactic valid C code with two two loops") {
    val e = nFun(n => nFun(m => fun(ArrayType(n, ArrayType(m, float)))(xs =>
      xs :>> mapSeq(mapSeq(fun(x => x))) :>> map(reorderWithStride(Cst(128)))
    )))

    val p = idealised.C.ProgramGenerator.makeCode(idealised.DPIA.FromSurfaceLanguage(TypeInference(e, Map())))
    val code = p.code
    SyntaxChecker(code)
    println(code)

    "for".r.findAllIn(code).length shouldBe 2
  }
}
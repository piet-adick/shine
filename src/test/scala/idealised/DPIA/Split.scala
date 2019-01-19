package idealised.DPIA

import idealised.SurfaceLanguage.DSL._
import idealised.SurfaceLanguage.Types._
import idealised.util.SyntaxChecker
import lift.arithmetic._

class Split extends idealised.Tests {

  test("Simple split example should generate syntactic valid C code with two for loops") {
    val slideExample = fun(ArrayType(SizeVar("N"), float))(xs => xs :>> split(2) :>> mapSeq(mapSeq(fun(x => x))))

    val p = idealised.C.ProgramGenerator.makeCode(TypeInference(slideExample, Map()).toPhrase)
    val code = p.code
    SyntaxChecker(code)
    println(code)

    "for".r.findAllIn(code).length shouldBe 2
  }

  test("Simple 2D split example with separate maps should generate syntactic valid OpenMP code with three for loops") {
    val slideExample = fun(ArrayType(SizeVar("N"), ArrayType(SizeVar("M"), float)))( xs =>
      xs :>> map(split(2)) :>> mapSeq(mapSeq(mapSeq(fun(x => x)))) )

    val p = idealised.OpenMP.ProgramGenerator.makeCode(TypeInference(slideExample, Map()).toPhrase)
    val code = p.code
    SyntaxChecker(code)
    println(code)

    "for".r.findAllIn(code).length shouldBe 3
  }

  test("Simple 2D split example with merged maps should generate syntactic valid OpenMP code with three for loops") {
    val slideExample = fun(ArrayType(SizeVar("N"), ArrayType(SizeVar("M"), float)))( xs =>
      xs :>> mapSeq(split(2) >>> mapSeq(mapSeq(fun(x => x)))) )

    val p = idealised.OpenMP.ProgramGenerator.makeCode(TypeInference(slideExample, Map()).toPhrase)
    val code = p.code
    SyntaxChecker(code)
    println(code)

    "for".r.findAllIn(code).length shouldBe 3
  }

}

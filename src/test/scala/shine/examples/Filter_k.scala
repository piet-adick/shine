package shine.examples

import shine.C.ProgramGenerator
import shine.DPIA.Phrases._
import shine.DPIA.Types._
import shine.DPIA.FunctionalPrimitives._
import shine.DPIA._

class Filter_k extends test_util.Tests {
  test("filter_k") {
    val arrayLength = NatIdentifier(freshName("arrayLength"))
    val array = Identifier(freshName("array"), ExpType(ArrayType(arrayLength, int), read))
    val element = Identifier(freshName("element"), ExpType(int, read))
    val condition = BinOp(Operators.Binary.EQ, BinOp(Operators.Binary.MOD, element, Literal(2)), Literal(0))

    val filter_k = DepLambda[NatKind](arrayLength)(Lambda[ExpType, ExpType](array,
      MapSeq(arrayLength, int, int,
        Lambda[ExpType, ExpType](element, IfThenElse(condition, element, Literal(0))),
        array)
    ))

    println(ProgramGenerator.makeCode(filter_k, "filter_k").code)
  }
}

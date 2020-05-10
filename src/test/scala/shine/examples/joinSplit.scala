package shine.examples

import shine.C.ProgramGenerator
import shine.DPIA.FunctionalPrimitives.{Drop, Fst, Join, MapSeq, ReduceSeq, Snd, Split}
import shine.DPIA.Phrases.{BinOp, Identifier, Lambda, Literal, Operators}
import shine.DPIA.Semantics.OperationalSemantics.FloatData
import shine.DPIA.Types.{ArrayType, ExpType, FunType, PairType, f32, read, write}
import shine.DPIA.freshName
import shine.test_util

//Hier sollen drei Testfälle beschrieben werden
class joinSplit extends test_util.Tests {


  //Nur die oberste Zeile wird der 2x2 Matrix addiert
  test("addiereNurdieZweiteSpalteMatrix") {
    val x = Identifier(freshName("x"), ExpType(f32, read))
    val y = Identifier(freshName("y"), ExpType(f32, read))
    val add = Lambda[ExpType, FunType[ExpType, ExpType]](x, Lambda[ExpType, ExpType](y, BinOp(Operators.Binary.ADD, x, y)))


    val matrix2x2 = Identifier(freshName("matrix2x2_"), ExpType(ArrayType(2, ArrayType(2, f32)), read))

    val addiereNurdieZweiteSpalteMatrix = Lambda[ExpType, ExpType](matrix2x2,
      BinOp(Operators.Binary.ADD,
        ReduceSeq(2, f32, f32, add, Literal(FloatData(0.0f)), Join(1, 2, read, f32, Drop(1, 1, read, ArrayType(2, f32), matrix2x2)
        )), Literal(FloatData(1.0f))))

    println(ProgramGenerator.makeCode(addiereNurdieZweiteSpalteMatrix, "addiereNurdieZweiteSpalteMatrix").code)
  }

  //calculate the determinante of a 2x2 matrix als Paare
  test("determinante2x2Pair") {
    val matrix2x2PairPair = Identifier(freshName("matrix2x2_"), ExpType(PairType(PairType(f32, f32), PairType(f32, f32)), read))

    val determinante2x2Pair = Lambda[ExpType, ExpType](
      matrix2x2PairPair,
      BinOp(Operators.Binary.SUB,
        //ad
        BinOp(Operators.Binary.MUL,
          //a
          Fst(f32, f32, Fst(PairType(f32, f32), PairType(f32, f32),
            matrix2x2PairPair
          )),
          //d
          Snd(f32, f32, Snd(PairType(f32, f32), PairType(f32, f32),
            matrix2x2PairPair
          ))),
        //bc
        BinOp(Operators.Binary.MUL,
          //a
          Fst(f32, f32, Snd(PairType(f32, f32), PairType(f32, f32),
            matrix2x2PairPair
          )),
          //d
          Snd(f32, f32, Fst(PairType(f32, f32), PairType(f32, f32),
            matrix2x2PairPair
          )))

      )

    )
    println(ProgramGenerator.makeCode(determinante2x2Pair, "determinante2x2Pair").code)
  }

  //Jedes Element der 2x2Matrix wird quadriert
  test("squareEveryElementInMatrix") {
    val x = Identifier(freshName("x"), ExpType(f32, read))
    val square = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, x, x))
    val matrix2x2 = Identifier(freshName("matrix2x2_"), ExpType(ArrayType(2, ArrayType(2, f32)), read))

    val squareEveryElementInMatrix = Lambda[ExpType, ExpType](
      matrix2x2,
      Split(2, 2, read, f32,
        MapSeq(4, f32, f32, square, Join(2, 2, write, f32,
          matrix2x2
        ))
      )
    )

    println(ProgramGenerator.makeCode(squareEveryElementInMatrix, "squareEveryElementInMatrix").code)
  }
}

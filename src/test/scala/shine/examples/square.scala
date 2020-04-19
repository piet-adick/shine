package shine.examples

import shine.C.ProgramGenerator
import shine.DPIA.FunctionalPrimitives.{Drop, Fst, Join, MapSeq, ReduceSeq, Snd, Split}
import shine.DPIA.Phrases.{BinOp, Identifier, Lambda, Literal, Operators}
import shine.DPIA.Semantics.OperationalSemantics.FloatData
import shine.DPIA.Types.{ArrayType, ExpType, FunType, PairType, float, read, write}
import shine.DPIA.freshName

//Hier sollen drei Testfälle beschrieben werden
  class diagonale extends test_util.Tests {


    //Nur die oberste Zeile wird der 2x2 Matrix addiert
    test("addiereNurErstenBeidenElementeMatrix") {
      val x = Identifier(freshName("x"), ExpType(float, read))
      val y = Identifier(freshName("y"), ExpType(float, read))
      val add = Lambda[ExpType, FunType[ExpType, ExpType]](x, Lambda[ExpType, ExpType](y, BinOp(Operators.Binary.ADD, x, y)))


      val matrix2x2 = Identifier(freshName("matrix2x2"), ExpType(ArrayType(2, ArrayType(2, float)), read))

      val addiereNurErstenBeidenElementeMatrix = Lambda[ExpType, ExpType](matrix2x2,
        BinOp(Operators.Binary.ADD,
          ReduceSeq(2, float, float, add, Literal(FloatData(0.0f)), Join(1, 2, read, float, Drop(1, 1, read, ArrayType(2, float), matrix2x2)
          )), Literal(FloatData(1.0f))))

      println(ProgramGenerator.makeCode(addiereNurErstenBeidenElementeMatrix, "addiereNurErstenBeidenElementeMatrix").code)
    }

    //calculate the determinante of a 2x2 matrix als Paare
    test("determinante2x2Pair") {
      val matrix2x2PairPair = Identifier(freshName("matrix2x2"), ExpType(PairType(PairType(float, float), PairType(float, float)), read))

      val determinante2x2Pair = Lambda[ExpType, ExpType](
        matrix2x2PairPair,
        BinOp(Operators.Binary.SUB,
          //ad
          BinOp(Operators.Binary.MUL,
            //a
            Fst(float, float, Fst(PairType(float, float), PairType(float, float),
              matrix2x2PairPair
            )),
            //d
            Snd(float, float, Snd(PairType(float, float), PairType(float, float),
              matrix2x2PairPair
            ))),
          //bc
          BinOp(Operators.Binary.MUL,
            //a
            Fst(float, float, Snd(PairType(float, float), PairType(float, float),
              matrix2x2PairPair
            )),
            //d
            Snd(float, float, Fst(PairType(float, float), PairType(float, float),
              matrix2x2PairPair
            )))

        )

      )
      println(ProgramGenerator.makeCode(determinante2x2Pair, "determinante2x2Pair").code)
    }

    //Jedes Element der 2x2Matrix wird quadriert
    test("squareEveryElementInMatrix") {
      val x = Identifier(freshName("x"), ExpType(float, read))
      val square = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, x, x))
      val matrix2x2 = Identifier(freshName("matrix2x2"), ExpType(ArrayType(2, ArrayType(2, float)), read))

      val squareEveryElementInMatrix = Lambda[ExpType, ExpType](
        matrix2x2,
        Split(2, 2, read, float,
          MapSeq(4, float, float, square, Join(2, 2, write, float,
            matrix2x2
          ))
        )
      )

      println(ProgramGenerator.makeCode(squareEveryElementInMatrix, "squareEveryElementInMatrix").code)
    }
  }

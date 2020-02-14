package shine.examples


import shine.C.ProgramGenerator
import shine.DPIA.FunctionalPrimitives.{Fst, MapSeq, ReduceSeq, Snd, Zip}
import shine.DPIA.Phrases.{BinOp, DepLambda, Identifier, IfThenElse, Lambda, Literal, Operators}
import shine.DPIA.Semantics.OperationalSemantics.FloatData
import shine.DPIA.Types.{ArrayType, ExpType, FunType, NatKind, PairType, float, read}
import shine.DPIA.{NatIdentifier, freshName}

//the amount of a vector should be compared with another vector and
//the greater vector should be returned
class amount_vector extends test_util.Tests{

  //calculate the square of the amount of a vector
  test("square of the amount_vector"){
    val n = NatIdentifier(freshName("n")) //size of the vector
    val vecA = Identifier(freshName("vecA"), ExpType(ArrayType(n, float), read))
//    val vecB = Identifier(freshName("vecB"), ExpType(ArrayType(n, float), read))

    val x = Identifier(freshName("x"), ExpType(float, read))
    val y = Identifier(freshName("y"), ExpType(float, read))
    val p = Identifier(freshName("p"), ExpType(PairType(float, float), read))

    val add = Lambda[ExpType, FunType[ExpType, ExpType]](x, Lambda[ExpType, ExpType](y, BinOp(Operators.Binary.ADD, x, y)))
    val mul = Lambda[ExpType, ExpType](p, BinOp(Operators.Binary.MUL, Fst(float, float, p), Snd(float, float, p)))

    val amount_vector = DepLambda[NatKind](n)(Lambda[ExpType, ExpType](
      vecA,
        ReduceSeq(n, float, float, add, Literal(FloatData(0.0f)),
        MapSeq(n, PairType(float,float), float, mul,
        Zip(n, float, float, vecA, vecA)))
    ))

    println(ProgramGenerator.makeCode(amount_vector, "amount_vector").code)

  }

  //returns 1 if the vector has the amount 1
  test("amount_vector_equals_1"){
    val n = NatIdentifier(freshName("n")) //size of the vector
    val vecA = Identifier(freshName("vecA"), ExpType(ArrayType(n, float), read))

    val x = Identifier(freshName("x"), ExpType(float, read))
    val y = Identifier(freshName("y"), ExpType(float, read))
    val p = Identifier(freshName("p"), ExpType(PairType(float, float), read))

    val add = Lambda[ExpType, FunType[ExpType, ExpType]](x, Lambda[ExpType, ExpType](y, BinOp(Operators.Binary.ADD, x, y)))
    val mul = Lambda[ExpType, ExpType](p, BinOp(Operators.Binary.MUL, Fst(float, float, p), Snd(float, float, p)))

    //val amount_vector = Lambda[ExpType, ExpType](
    //  vecA,
    //ReduceSeq(n, float, float, add, Literal(FloatData(0.0f)),
    //    MapSeq(n, PairType(float,float), float, mul,
    //      Zip(n, float, float, vecA, vecA)))
    //)

    //val a = Identifier(freshName("a"), ExpType(float, read))
    //val b = Identifier(freshName("b"), ExpType(float, read))
    //val max = Lambda[ExpType, FunType[ExpType, ExpType]](a, Lambda[ExpType, ExpType](b, IfThenElse(BinOp(Operators.Binary.GT, a,b), a, b)))

    val amount_vector_equals_1 = DepLambda[NatKind](n)(
        Lambda[ExpType, ExpType](vecA,
          IfThenElse(BinOp(Operators.Binary.EQ,
            ReduceSeq(n, float, float, add, Literal(FloatData(0.0f)),
            MapSeq(n, PairType(float,float), float, mul,
              Zip(n, float, float, vecA, vecA))),
            Literal(FloatData(1.0f))), Literal(FloatData(1.0f)), Literal(FloatData(0.0f)))
        )
    )

    println(ProgramGenerator.makeCode(amount_vector_equals_1, "amount_vector_equals_1").code)

  }

  //this test failes and I don't know why, because I only switch the
  //arguments of the test above, which works perfectly
  //returns 1 if the vector has the amount 1
  test("amount_vector_equals_1_secondversion"){
    val n = NatIdentifier(freshName("n")) //size of the vector
    val vecA = Identifier(freshName("vecA"), ExpType(ArrayType(n, float), read))

    val x = Identifier(freshName("x"), ExpType(float, read))
    val y = Identifier(freshName("y"), ExpType(float, read))
    val p = Identifier(freshName("p"), ExpType(PairType(float, float), read))

    val add = Lambda[ExpType, FunType[ExpType, ExpType]](x, Lambda[ExpType, ExpType](y, BinOp(Operators.Binary.ADD, x, y)))
    val mul = Lambda[ExpType, ExpType](p, BinOp(Operators.Binary.MUL, Fst(float, float, p), Snd(float, float, p)))

    //val amount_vector = Lambda[ExpType, ExpType](
    //  vecA,
    //ReduceSeq(n, float, float, add, Literal(FloatData(0.0f)),
    //    MapSeq(n, PairType(float,float), float, mul,
    //      Zip(n, float, float, vecA, vecA)))
    //)

    //val a = Identifier(freshName("a"), ExpType(float, read))
    //val b = Identifier(freshName("b"), ExpType(float, read))
    //val max = Lambda[ExpType, FunType[ExpType, ExpType]](a, Lambda[ExpType, ExpType](b, IfThenElse(BinOp(Operators.Binary.GT, a,b), a, b)))

    val amount_vector_equals_1 = DepLambda[NatKind](n)(
      Lambda[ExpType, ExpType](vecA,
        IfThenElse(BinOp(Operators.Binary.EQ,
          Literal(FloatData(1.0f)),
          ReduceSeq(n, float, float, add, Literal(FloatData(0.0f)),
            MapSeq(n, PairType(float,float), float, mul,
              Zip(n, float, float, vecA, vecA)))), Literal(FloatData(1.0f)), Literal(FloatData(0.0f)))
      )
    )

    println(ProgramGenerator.makeCode(amount_vector_equals_1, "amount_vector_equals_1").code)

  }

  //returns 1 if the vector have the same amount
  /*test("same_amount_vector"){
    val n = NatIdentifier(freshName("n")) //size of the vector
    val vecA = Identifier(freshName("vecA"), ExpType(ArrayType(n, float), read))
    val vecB = Identifier(freshName("vecB"), ExpType(ArrayType(n, float), read))

    val x = Identifier(freshName("x"), ExpType(float, read))
    val y = Identifier(freshName("y"), ExpType(float, read))
    val p = Identifier(freshName("p"), ExpType(PairType(float, float), read))

    val add = Lambda[ExpType, FunType[ExpType, ExpType]](x, Lambda[ExpType, ExpType](y, BinOp(Operators.Binary.ADD, x, y)))
    val mul = Lambda[ExpType, ExpType](p, BinOp(Operators.Binary.MUL, Fst(float, float, p), Snd(float, float, p)))

    //val amount_vector = Lambda[ExpType, ExpType](
    //  vecA,
    //ReduceSeq(n, float, float, add, Literal(FloatData(0.0f)),
    //    MapSeq(n, PairType(float,float), float, mul,
    //      Zip(n, float, float, vecA, vecA)))
    //)

    //val a = Identifier(freshName("a"), ExpType(float, read))
    //val b = Identifier(freshName("b"), ExpType(float, read))
    //val max = Lambda[ExpType, FunType[ExpType, ExpType]](a, Lambda[ExpType, ExpType](b, IfThenElse(BinOp(Operators.Binary.GT, a,b), a, b)))

    val same_amount_vector = DepLambda[NatKind](n)(
      Lambda[ExpType, FunType[ExpType, ExpType]](vecA,
        Lambda[ExpType, ExpType](vecB,
          IfThenElse(BinOp(Operators.Binary.EQ,
            Literal(FloatData(0.0f)),
            ReduceSeq(n, float, float, add, Literal(FloatData(0.0f)),
              MapSeq(n, PairType(float,float), float, mul,
                Zip(n, float, float, vecA, vecA)))), Literal(FloatData(1.0f)), Literal(FloatData(0.0f)))
        ))
    )

    println(ProgramGenerator.makeCode(same_amount_vector, "same_amount_vector").code)

  }*/
}

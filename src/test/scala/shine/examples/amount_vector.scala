package shine.examples


import shine.C.ProgramGenerator
import shine.DPIA.FunctionalPrimitives.{Fst, MapSeq, ReduceSeq, Snd, Zip}
import shine.DPIA.Phrases.{BinOp, DepLambda, Identifier, IfThenElse, Lambda, Literal, Operators}
import shine.DPIA.Semantics.OperationalSemantics.FloatData
import shine.DPIA.Types.{ArrayType, ExpType, FunType, NatKind, PairType, f32, read}
import shine.DPIA.{NatIdentifier, freshName}
import shine.test_util

//the amount of a vector should be compared with another vector and
//the greater vector should be returned
class amount_vector extends test_util.Tests{

  //calculate the square of the amount of a vector
  test("square of the amount_vector"){
    val n = NatIdentifier(freshName("n")) //size of the vector
    val vecA = Identifier(freshName("vecA"), ExpType(ArrayType(n, f32), read))
//    val vecB = Identifier(freshName("vecB"), ExpType(ArrayType(n, f32), read))

    val x = Identifier(freshName("x"), ExpType(f32, read))
    val y = Identifier(freshName("y"), ExpType(f32, read))
    val p = Identifier(freshName("p"), ExpType(PairType(f32, f32), read))

    val add = Lambda[ExpType, FunType[ExpType, ExpType]](x, Lambda[ExpType, ExpType](y, BinOp(Operators.Binary.ADD, x, y)))
    val mul = Lambda[ExpType, ExpType](p, BinOp(Operators.Binary.MUL, Fst(f32, f32, p), Snd(f32, f32, p)))

    val amount_vector = DepLambda[NatKind](n)(Lambda[ExpType, ExpType](
      vecA,
        ReduceSeq(n, f32, f32, add, Literal(FloatData(0.0f)),
        MapSeq(n, PairType(f32,f32), f32, mul,
        Zip(n, f32, f32, vecA, vecA)))
    ))

    println(ProgramGenerator.makeCode(amount_vector, "amount_vector").code)

  }

  //returns 1 if the vector has the amount 1
  test("amount_vector_equals_1"){
    val n = NatIdentifier(freshName("n")) //size of the vector
    val vecA = Identifier(freshName("vecA"), ExpType(ArrayType(n, f32), read))

    val x = Identifier(freshName("x"), ExpType(f32, read))
    val y = Identifier(freshName("y"), ExpType(f32, read))
    val p = Identifier(freshName("p"), ExpType(PairType(f32, f32), read))

    val add = Lambda[ExpType, FunType[ExpType, ExpType]](x, Lambda[ExpType, ExpType](y, BinOp(Operators.Binary.ADD, x, y)))
    val mul = Lambda[ExpType, ExpType](p, BinOp(Operators.Binary.MUL, Fst(f32, f32, p), Snd(f32, f32, p)))

    //val amount_vector = Lambda[ExpType, ExpType](
    //  vecA,
    //ReduceSeq(n, f32, f32, add, Literal(FloatData(0.0f)),
    //    MapSeq(n, PairType(f32,f32), f32, mul,
    //      Zip(n, f32, f32, vecA, vecA)))
    //)

    //val a = Identifier(freshName("a"), ExpType(f32, read))
    //val b = Identifier(freshName("b"), ExpType(f32, read))
    //val max = Lambda[ExpType, FunType[ExpType, ExpType]](a, Lambda[ExpType, ExpType](b, IfThenElse(BinOp(Operators.Binary.GT, a,b), a, b)))

    val amount_vector_equals_1 = DepLambda[NatKind](n)(
        Lambda[ExpType, ExpType](vecA,
          IfThenElse(BinOp(Operators.Binary.EQ,
            ReduceSeq(n, f32, f32, add, Literal(FloatData(0.0f)),
            MapSeq(n, PairType(f32,f32), f32, mul,
              Zip(n, f32, f32, vecA, vecA))),
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
    val vecA = Identifier(freshName("vecA"), ExpType(ArrayType(n, f32), read))

    val x = Identifier(freshName("x"), ExpType(f32, read))
    val y = Identifier(freshName("y"), ExpType(f32, read))
    val p = Identifier(freshName("p"), ExpType(PairType(f32, f32), read))

    val add = Lambda[ExpType, FunType[ExpType, ExpType]](x, Lambda[ExpType, ExpType](y, BinOp(Operators.Binary.ADD, x, y)))
    val mul = Lambda[ExpType, ExpType](p, BinOp(Operators.Binary.MUL, Fst(f32, f32, p), Snd(f32, f32, p)))

    //val amount_vector = Lambda[ExpType, ExpType](
    //  vecA,
    //ReduceSeq(n, f32, f32, add, Literal(FloatData(0.0f)),
    //    MapSeq(n, PairType(f32,f32), f32, mul,
    //      Zip(n, f32, f32, vecA, vecA)))
    //)

    //val a = Identifier(freshName("a"), ExpType(f32, read))
    //val b = Identifier(freshName("b"), ExpType(f32, read))
    //val max = Lambda[ExpType, FunType[ExpType, ExpType]](a, Lambda[ExpType, ExpType](b, IfThenElse(BinOp(Operators.Binary.GT, a,b), a, b)))

    val amount_vector_equals_1 = DepLambda[NatKind](n)(
      Lambda[ExpType, ExpType](vecA,
        IfThenElse(BinOp(Operators.Binary.EQ,
          Literal(FloatData(1.0f)),
          ReduceSeq(n, f32, f32, add, Literal(FloatData(0.0f)),
            MapSeq(n, PairType(f32,f32), f32, mul,
              Zip(n, f32, f32, vecA, vecA)))), Literal(FloatData(1.0f)), Literal(FloatData(0.0f)))
      )
    )

    println(ProgramGenerator.makeCode(amount_vector_equals_1, "amount_vector_equals_1").code)

  }

  //returns 1 if the vector have the same amount
  /*test("same_amount_vector"){
    val n = NatIdentifier(freshName("n")) //size of the vector
    val vecA = Identifier(freshName("vecA"), ExpType(ArrayType(n, f32), read))
    val vecB = Identifier(freshName("vecB"), ExpType(ArrayType(n, f32), read))

    val x = Identifier(freshName("x"), ExpType(f32, read))
    val y = Identifier(freshName("y"), ExpType(f32, read))
    val p = Identifier(freshName("p"), ExpType(PairType(f32, f32), read))

    val add = Lambda[ExpType, FunType[ExpType, ExpType]](x, Lambda[ExpType, ExpType](y, BinOp(Operators.Binary.ADD, x, y)))
    val mul = Lambda[ExpType, ExpType](p, BinOp(Operators.Binary.MUL, Fst(f32, f32, p), Snd(f32, f32, p)))

    //val amount_vector = Lambda[ExpType, ExpType](
    //  vecA,
    //ReduceSeq(n, f32, f32, add, Literal(FloatData(0.0f)),
    //    MapSeq(n, PairType(f32,f32), f32, mul,
    //      Zip(n, f32, f32, vecA, vecA)))
    //)

    //val a = Identifier(freshName("a"), ExpType(f32, read))
    //val b = Identifier(freshName("b"), ExpType(f32, read))
    //val max = Lambda[ExpType, FunType[ExpType, ExpType]](a, Lambda[ExpType, ExpType](b, IfThenElse(BinOp(Operators.Binary.GT, a,b), a, b)))

    val same_amount_vector = DepLambda[NatKind](n)(
      Lambda[ExpType, FunType[ExpType, ExpType]](vecA,
        Lambda[ExpType, ExpType](vecB,
          IfThenElse(BinOp(Operators.Binary.EQ,
            Literal(FloatData(0.0f)),
            ReduceSeq(n, f32, f32, add, Literal(FloatData(0.0f)),
              MapSeq(n, PairType(f32,f32), f32, mul,
                Zip(n, f32, f32, vecA, vecA)))), Literal(FloatData(1.0f)), Literal(FloatData(0.0f)))
        ))
    )

    println(ProgramGenerator.makeCode(same_amount_vector, "same_amount_vector").code)

  }*/
}

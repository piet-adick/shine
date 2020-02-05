package shine.examples

import shine.C.ProgramGenerator
import shine.DPIA.Phrases._
import shine.DPIA.Types._
//import shine.DPIA.Types.AddressSpace._
import shine.DPIA.FunctionalPrimitives._
import shine.DPIA.Semantics.OperationalSemantics.FloatData
import shine.DPIA._
//import shine.OpenCL.FunctionalPrimitives.{MapGlobal, OpenCLReduceSeq, To}
//import shine.OpenCL.KernelGenerator
//import shine.examples.dotProduct

class matrixmultiplication extends test_util.Tests {

  test("matrixMult") {


    // def dotproduct = fun((v,w) => reduce(+, 0) <<: map(*) <<: zip(v,w)) // rise
    // def matrixmultiplication = fun((v,w)) => map(map(dotproduct(v,transpose(w)))  => transpose(w)

    //def add(a: float, b: float):float = a + b

    val x = Identifier(freshName("x"), ExpType(PairType(float, float), read))
    val y = Identifier(freshName("y"), ExpType(float, read))
    val z = Identifier(freshName("z"), ExpType(float, read))

    val n = NatIdentifier(freshName("n"))
    //val vecA = Identifier(freshName("vecA"), ExpType(ArrayType(n, float), read))
    //val vecB = Identifier(freshName("vecB"), ExpType(ArrayType(n, float), read))
    val columnB = Identifier(freshName("columnB"), ExpType(ArrayType(n, float), read))
    val rowA = Identifier(freshName("rowA"), ExpType(ArrayType(n, float), read))
    val matrixA = Identifier(freshName("MatrixA"), ExpType(ArrayType(n, ArrayType(n,float)),read))
    val matrixB = Identifier(freshName("MatrixA"), ExpType(ArrayType(n, ArrayType(n,float)),read))

    val mul = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, Fst(float, float, x), Snd(float, float, x)))
    val add = Lambda[ExpType, FunType[ExpType, ExpType]](y, Lambda[ExpType, ExpType](z, BinOp(Operators.Binary.ADD, y, z)))

    //val dot = DepLambda[NatKind](n)(Lambda[ExpType, FunType[ExpType, ExpType]](vecA, Lambda[ExpType, ExpType](vecB,
      //ReduceSeq(n, float, float, add, Literal(FloatData(0.0f)),
        //MapSeq(n, PairType(float, float), float, mul,
          //Zip(n, float, float, vecA, vecB))))))
    val dotproduct = Lambda[ExpType, ExpType](rowA,
        ReduceSeq(n, float, float, add, Literal(FloatData(0.0f)),
          MapSeq(n, PairType(float, float), float, mul,
            Zip(n, float, float, rowA, columnB))))

    //val vectorPair = Identifier(freshName("vectorPair"), ExpType(PairType(ArrayType(n,float), ArrayType(n,float)), read))
    //val dotSeq = Lambda[ExpType, ExpType](vectorPair, dot)
    val matrixMult = DepLambda[NatKind](n)(Lambda[ExpType, FunType[ExpType, ExpType]](
      matrixA, Lambda[ExpType, ExpType](matrixB,
        MapSeq(n, ArrayType(n,float), ArrayType(n,float),
          Lambda[ExpType, ExpType](columnB,
            MapSeq(n, ArrayType(n,float), float,
              dotproduct,
              Transpose(n, n, float,matrixB))),
          matrixA))
    ))
    println("matrixMult:\n " + ProgramGenerator.makeCode(matrixMult, "matrixMult").code)

    /*
    println(add(1,2))
    add(8.0f,2.0f) shouldBe 10
    */
  }


}
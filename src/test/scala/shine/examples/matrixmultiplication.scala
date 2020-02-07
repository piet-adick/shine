package shine.examples

import shine.C.ProgramGenerator
import shine.DPIA.Phrases._
import shine.DPIA.Types._
import shine.DPIA.Types.AddressSpace._
import shine.DPIA.FunctionalPrimitives._
import shine.DPIA.Semantics.OperationalSemantics.FloatData
import shine.DPIA._
import shine.OpenCL.FunctionalPrimitives.{MapGlobal, OpenCLReduceSeq, To}
import shine.OpenCL.KernelGenerator

class matrixmultiplication extends test_util.Tests {

  test("matrixMult") {


    // def dotproduct = fun((v,w) => reduce(+, 0) <<: map(*) <<: zip(v,w)) // rise
    // def matrixmultiplication = fun((v,w)) => map(map(dotproduct(v,transpose(w)))  => transpose(w)

    //def add(a: float, b: float):float = a + b

    val x = Identifier(freshName("x"), ExpType(PairType(float, float), read))
    val y = Identifier(freshName("y"), ExpType(float, read))
    val z = Identifier(freshName("z"), ExpType(float, read))

    val n = NatIdentifier(freshName("n"))
    val columnB = Identifier(freshName("columnB"), ExpType(ArrayType(n, float), read))
    val rowA = Identifier(freshName("rowA"), ExpType(ArrayType(n, float), read))
    val matrixA = Identifier(freshName("MatrixA"), ExpType(ArrayType(n, ArrayType(n,float)),read))
    val matrixB = Identifier(freshName("MatrixB"), ExpType(ArrayType(n, ArrayType(n,float)),read))

    val mul = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, Fst(float, float, x), Snd(float, float, x)))
    val add = Lambda[ExpType, FunType[ExpType, ExpType]](y, Lambda[ExpType, ExpType](z, BinOp(Operators.Binary.ADD, y, z)))

    val dotproduct = Lambda[ExpType, ExpType](rowA,
        ReduceSeq(n, float, float, add, Literal(FloatData(0.0f)),
          MapSeq(n, PairType(float, float), float, mul,
            Zip(n, float, float, rowA, columnB))))

    val matrixMult = DepLambda[NatKind](n)(Lambda[ExpType, FunType[ExpType, ExpType]](
      matrixA, Lambda[ExpType, ExpType](matrixB,
        MapSeq(n, ArrayType(n,float), ArrayType(n,float),
          Lambda[ExpType, ExpType](columnB,
            MapSeq(n, ArrayType(n,float), float,
              dotproduct,
              Transpose(n, n, float,matrixB))),
          matrixA))
    ))
    println(ProgramGenerator.makeCode(matrixMult, "matrixMult").code)

  }

  test("matrixMult OpenCl") {


    // def dotproduct = fun((v,w) => reduce(+, 0) <<: map(*) <<: zip(v,w)) // rise
    // def matrixmultiplication = fun((v,w)) => map(map(dotproduct(v,transpose(w)))  => transpose(w)

    //def add(a: float, b: float):float = a + b

    val x = Identifier(freshName("x"), ExpType(PairType(float, float), read))
    val y = Identifier(freshName("y"), ExpType(float, read))
    val z = Identifier(freshName("z"), ExpType(float, read))

    val n = NatIdentifier(freshName("n"))
    val columnB = Identifier(freshName("columnB"), ExpType(ArrayType(n, float), read))
    val rowA = Identifier(freshName("rowA"), ExpType(ArrayType(n, float), read))
    val matrixA = Identifier(freshName("MatrixA"), ExpType(ArrayType(n, ArrayType(n,float)),read))
    val matrixB = Identifier(freshName("MatrixA"), ExpType(ArrayType(n, ArrayType(n,float)),read))

    val mul = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, Fst(float, float, x), Snd(float, float, x)))
    val add = Lambda[ExpType, FunType[ExpType, ExpType]](y, Lambda[ExpType, ExpType](z, BinOp(Operators.Binary.ADD, y, z)))

    val dotproduct = Lambda[ExpType, ExpType](rowA,
      OpenCLReduceSeq(n, Global, float, float, add, Literal(FloatData(0.0f)),
        To(Global, ArrayType(n, float),
        MapGlobal(0)(n, PairType(float, float), float, mul,
          Zip(n, float, float, rowA, columnB))), false))

    val matrixMult = DepLambda[NatKind](n)(Lambda[ExpType, FunType[ExpType, ExpType]](
      matrixA, Lambda[ExpType, ExpType](matrixB,
        MapGlobal(0)(n, ArrayType(n,float), ArrayType(n,float),
          Lambda[ExpType, ExpType](columnB,
            MapGlobal(0)(n, ArrayType(n,float), float,
              dotproduct,
              Transpose(n, n, float,matrixB))),
          matrixA))
    ))
    println(KernelGenerator.makeCode(matrixMult, "matrixMult_OpenCl").code)

  }


}
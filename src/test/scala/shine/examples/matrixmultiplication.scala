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
import shine.test_util

class matrixmultiplication extends test_util.Tests {

  test("matrixMult") {
    // def dotproduct = fun((v,w) => reduce(+, 0) <<: map(*) <<: zip(v,w)) // rise
    // def matrixmultiplication = fun((v,w)) => map(map(dotproduct(v,transpose(w)))  => transpose(w)

    //def add(a: f32, b: f32):f32 = a + b

    val x = Identifier(freshName("x"), ExpType(PairType(f32, f32), read))
    val y = Identifier(freshName("y"), ExpType(f32, read))
    val z = Identifier(freshName("z"), ExpType(f32, read))

    val n = NatIdentifier(freshName("n"))
    val columnB = Identifier(freshName("columnB"), ExpType(ArrayType(n, f32), read))
    val rowA = Identifier(freshName("rowA"), ExpType(ArrayType(n, f32), read))
    val matrixA = Identifier(freshName("MatrixA"), ExpType(ArrayType(n, ArrayType(n,f32)),read))
    val matrixB = Identifier(freshName("MatrixB"), ExpType(ArrayType(n, ArrayType(n,f32)),read))

    val mul = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, Fst(f32, f32, x), Snd(f32, f32, x)))
    val add = Lambda[ExpType, FunType[ExpType, ExpType]](y, Lambda[ExpType, ExpType](z, BinOp(Operators.Binary.ADD, y, z)))

    val dotproduct = Lambda[ExpType, ExpType](rowA,
        ReduceSeq(n, f32, f32, add, Literal(FloatData(0.0f)),
          MapSeq(n, PairType(f32, f32), f32, mul,
            Zip(n, f32, f32, rowA, columnB))))

    val matrixMult = DepLambda[NatKind](n)(Lambda[ExpType, FunType[ExpType, ExpType]](
      matrixA, Lambda[ExpType, ExpType](matrixB,
        MapSeq(n, ArrayType(n,f32), ArrayType(n,f32),
          Lambda[ExpType, ExpType](columnB,
            MapSeq(n, ArrayType(n,f32), f32,
              dotproduct,
              Transpose(n, n, f32,matrixB))),
          matrixA))
    ))
    println(ProgramGenerator.makeCode(matrixMult, "matrixMult").code)

  }

  test("matrixMult OpenCl") {


    // def dotproduct = fun((v,w) => reduce(+, 0) <<: map(*) <<: zip(v,w)) // rise
    // def matrixmultiplication = fun((v,w)) => map(map(dotproduct(v,transpose(w)))  => transpose(w)

    //def add(a: f32, b: f32):f32 = a + b

    val x = Identifier(freshName("x"), ExpType(PairType(f32, f32), read))
    val y = Identifier(freshName("y"), ExpType(f32, read))
    val z = Identifier(freshName("z"), ExpType(f32, read))

    val n = NatIdentifier(freshName("n"))
    val columnB = Identifier(freshName("columnB"), ExpType(ArrayType(n, f32), read))
    val rowA = Identifier(freshName("rowA"), ExpType(ArrayType(n, f32), read))
    val matrixA = Identifier(freshName("MatrixA"), ExpType(ArrayType(n, ArrayType(n,f32)),read))
    val matrixB = Identifier(freshName("MatrixA"), ExpType(ArrayType(n, ArrayType(n,f32)),read))

    val mul = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, Fst(f32, f32, x), Snd(f32, f32, x)))
    val add = Lambda[ExpType, FunType[ExpType, ExpType]](y, Lambda[ExpType, ExpType](z, BinOp(Operators.Binary.ADD, y, z)))

    val dotproduct = Lambda[ExpType, ExpType](rowA,
      OpenCLReduceSeq(n, Global, f32, f32, add, Literal(FloatData(0.0f)),
        To(Global, ArrayType(n, f32),
        MapGlobal(0)(n, PairType(f32, f32), f32, mul,
          Zip(n, f32, f32, rowA, columnB))), false))

    val matrixMult = DepLambda[NatKind](n)(Lambda[ExpType, FunType[ExpType, ExpType]](
      matrixA, Lambda[ExpType, ExpType](matrixB,
        MapGlobal(0)(n, ArrayType(n,f32), ArrayType(n,f32),
          Lambda[ExpType, ExpType](columnB,
            MapGlobal(0)(n, ArrayType(n,f32), f32,
              dotproduct,
              Transpose(n, n, f32,matrixB))),
          matrixA))
    ))
    println(KernelGenerator.apply().makeCode(matrixMult, "matrixMult_OpenCl").code)

  }


}
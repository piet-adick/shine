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

class matrixmultiplication_nxm_mxr extends test_util.Tests {

  test("matrixMult") {


    // def dotproduct = fun((v,w) => reduce(+, 0) <<: map(*) <<: zip(v,w)) // rise
    // def matrixmultiplication = fun((v,w)) => map(map(dotproduct(v,transpose(w)))  => transpose(w)

    //def add(a: float, b: float):float = a + b

    val x = Identifier(freshName("x"), ExpType(PairType(float, float), read))
    val y = Identifier(freshName("y"), ExpType(float, read))
    val z = Identifier(freshName("z"), ExpType(float, read))

    val n = NatIdentifier(freshName("n"))
    val m = NatIdentifier(freshName("m"))
    val r = NatIdentifier(freshName("r"))
    val columnB = Identifier(freshName("columnB"), ExpType(ArrayType(m, float), read))
    val rowA = Identifier(freshName("rowA"), ExpType(ArrayType(m, float), read))
    val matrixA = Identifier(freshName("MatrixA"), ExpType(ArrayType(n, ArrayType(m,float)),read))
    val matrixB = Identifier(freshName("MatrixB"), ExpType(ArrayType(m, ArrayType(r,float)),read))

    val mul = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, Fst(float, float, x), Snd(float, float, x)))
    val add = Lambda[ExpType, FunType[ExpType, ExpType]](y, Lambda[ExpType, ExpType](z, BinOp(Operators.Binary.ADD, y, z)))

    val dotproduct = Lambda[ExpType, ExpType](rowA,
      ReduceSeq(m, float, float, add, Literal(FloatData(0.0f)),
        MapSeq(m, PairType(float, float), float, mul,
          Zip(m, float, float, rowA, columnB))))

    val matrixMult = DepLambda[NatKind](r)(DepLambda[NatKind](m)(DepLambda[NatKind](n)(
      Lambda[ExpType, FunType[ExpType, ExpType]](
      matrixA, Lambda[ExpType, ExpType](matrixB,
        MapSeq(n, ArrayType(m,float), ArrayType(r,float),
          Lambda[ExpType, ExpType](columnB,
            MapSeq(r, ArrayType(m,float), float,
              dotproduct,
              Transpose(m, r, float,matrixB))),
          matrixA))
    ))))
    println(ProgramGenerator.makeCode(matrixMult, "matrixMult").code)

  }

  test("gemm") {
    val x = Identifier(freshName("x"), ExpType(PairType(float, float), read))
    val y = Identifier(freshName("y"), ExpType(float, read))
    val z = Identifier(freshName("z"), ExpType(float, read))

    val n = NatIdentifier(freshName("n"))
    val m = NatIdentifier(freshName("m"))
    val k = NatIdentifier(freshName("k"))

    val rowAC = Identifier(freshName("vecAC"), ExpType(PairType(ArrayType(m, float),ArrayType(k, float)), read))
    val columnBFC = Identifier(freshName("columnBFC"), ExpType(PairType(ArrayType(m, float),float), read))
    val matA = Identifier(freshName("matA"), ExpType(ArrayType(n, ArrayType(m, float)), read))
    val matB = Identifier(freshName("matB"), ExpType(ArrayType(m, ArrayType(k, float)), read))
    val matC = Identifier(freshName("matC"), ExpType(ArrayType(n, ArrayType(k, float)), read))

    val mul = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, Fst(float, float, x), Snd(float, float, x)))
    val add = Lambda[ExpType, FunType[ExpType, ExpType]](y, Lambda[ExpType, ExpType](z, BinOp(Operators.Binary.ADD, y, z)))

    val gemm = DepLambda[NatKind](n)(DepLambda[NatKind](m)(DepLambda[NatKind](k)(
      Lambda[ExpType, FunType[ExpType, FunType[ExpType, ExpType]]](matC,
        Lambda[ExpType, FunType[ExpType, ExpType]](matA,
          Lambda[ExpType, ExpType](matB,
            MapSeq(n, PairType(ArrayType(m, float), ArrayType(k, float)), ArrayType(k, float),
              Lambda[ExpType, ExpType] (rowAC,
                MapSeq(k, PairType(ArrayType(m, float), float), float,
                  Lambda[ExpType, ExpType] (columnBFC,
                    BinOp(Operators.Binary.ADD,
                      ReduceSeq(m, float, float, add, Literal(FloatData(0.0f)),
                        MapSeq(m, PairType(float, float), float, mul,
                          Zip(m, float, float,
                            Fst(ArrayType(m,float), ArrayType(k, float), rowAC),
                            Fst(ArrayType(m,float), float, columnBFC)))),
                      Snd(ArrayType(m,float), float, columnBFC))),
                  Zip(k, ArrayType(m, float), float,
                    Transpose(m, k, float, matB), Snd(ArrayType(m,float), ArrayType(k, float), rowAC)))),
              Zip(n, ArrayType(m, float), ArrayType(k, float), matA, matC))))))))

    println(ProgramGenerator.makeCode(gemm, "gemm").code)
  }

  test("matrixMult OpenCl") {


    // def dotproduct = fun((v,w) => reduce(+, 0) <<: map(*) <<: zip(v,w)) // rise
    // def matrixmultiplication = fun((v,w)) => map(map(dotproduct(v,transpose(w)))  => transpose(w)

    //def add(a: float, b: float):float = a + b

    val x = Identifier(freshName("x"), ExpType(PairType(float, float), read))
    val y = Identifier(freshName("y"), ExpType(float, read))
    val z = Identifier(freshName("z"), ExpType(float, read))

    val n = NatIdentifier(freshName("n"))
    val m = NatIdentifier(freshName("m"))
    val r = NatIdentifier(freshName("r"))
    val columnB = Identifier(freshName("columnB"), ExpType(ArrayType(m, float), read))
    val rowA = Identifier(freshName("rowA"), ExpType(ArrayType(m, float), read))
    val matrixA = Identifier(freshName("MatrixA"), ExpType(ArrayType(n, ArrayType(m,float)),read))
    val matrixB = Identifier(freshName("MatrixB"), ExpType(ArrayType(m, ArrayType(r,float)),read))

    val mul = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, Fst(float, float, x), Snd(float, float, x)))
    val add = Lambda[ExpType, FunType[ExpType, ExpType]](y, Lambda[ExpType, ExpType](z, BinOp(Operators.Binary.ADD, y, z)))

    val dotproduct = Lambda[ExpType, ExpType](rowA,
      OpenCLReduceSeq(m, Global, float, float, add, Literal(FloatData(0.0f)),
        To(Global, ArrayType(m, float),
          MapGlobal(0)(m, PairType(float, float), float, mul,
            Zip(m, float, float, rowA, columnB))), false))

    val matrixMult = DepLambda[NatKind](r)(DepLambda[NatKind](m)(DepLambda[NatKind](n)(
      Lambda[ExpType, FunType[ExpType, ExpType]](
        matrixA, Lambda[ExpType, ExpType](matrixB,
          MapGlobal(0)(n, ArrayType(m,float), ArrayType(r,float),
            Lambda[ExpType, ExpType](columnB,
              MapGlobal(0)(r, ArrayType(m,float), float,
                dotproduct,
                Transpose(m, r, float,matrixB))),
            matrixA))
      ))))
    println(KernelGenerator.makeCode(matrixMult, "matrixMult").code)

  }
}
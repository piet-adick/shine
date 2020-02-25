package shine.examples

import shine.C.ProgramGenerator
import shine.DPIA.Phrases._
import shine.DPIA.Types._
import shine.DPIA.FunctionalPrimitives._
import shine.DPIA.Semantics.OperationalSemantics.FloatData
import shine.DPIA._
import shine.OpenCL.FunctionalPrimitives.{OpenCLReduceSeq, To}
import shine.OpenCL._
import shine.cuda.primitives.functional.MapThreads
import shine.test_util

class matrixmultiplication_nxm_mxr extends test_util.Tests {

  test("matrixMult") {


    // def dotproduct = fun((v,w) => reduce(+, 0) <<: map(*) <<: zip(v,w)) // rise
    // def matrixmultiplication = fun((v,w)) => map(map(dotproduct(v,transpose(w)))  => transpose(w)

    //def add(a: f32, b: f32):f32 = a + b

    val x = Identifier(freshName("x"), ExpType(PairType(f32, f32), read))
    val y = Identifier(freshName("y"), ExpType(f32, read))
    val z = Identifier(freshName("z"), ExpType(f32, read))

    val n = NatIdentifier(freshName("n"))
    val m = NatIdentifier(freshName("m"))
    val r = NatIdentifier(freshName("r"))
    val columnB = Identifier(freshName("columnB"), ExpType(ArrayType(m, f32), read))
    val rowA = Identifier(freshName("rowA"), ExpType(ArrayType(m, f32), read))
    val matrixA = Identifier(freshName("MatrixA"), ExpType(ArrayType(n, ArrayType(m,f32)),read))
    val matrixB = Identifier(freshName("MatrixB"), ExpType(ArrayType(m, ArrayType(r,f32)),read))

    val mul = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, Fst(f32, f32, x), Snd(f32, f32, x)))
    val add = Lambda[ExpType, FunType[ExpType, ExpType]](y, Lambda[ExpType, ExpType](z, BinOp(Operators.Binary.ADD, y, z)))

    val dotproduct = Lambda[ExpType, ExpType](rowA,
      ReduceSeq(m, f32, f32, add, Literal(FloatData(0.0f)),
        MapSeq(m, PairType(f32, f32), f32, mul,
          Zip(m, f32, f32, rowA, columnB))))

    val matrixMult = DepLambda[NatKind](r)(DepLambda[NatKind](m)(DepLambda[NatKind](n)(
      Lambda[ExpType, FunType[ExpType, ExpType]](
      matrixA, Lambda[ExpType, ExpType](matrixB,
        MapSeq(n, ArrayType(m,f32), ArrayType(r,f32),
          Lambda[ExpType, ExpType](columnB,
            MapSeq(r, ArrayType(m,f32), f32,
              dotproduct,
              Transpose(m, r, f32,matrixB))),
          matrixA))
    ))))
    println(ProgramGenerator.makeCode(matrixMult, "matrixMult").code)

  }

  test("gemm") {
    val x = Identifier(freshName("x"), ExpType(PairType(f32, f32), read))
    val y = Identifier(freshName("y"), ExpType(f32, read))
    val z = Identifier(freshName("z"), ExpType(f32, read))

    val n = NatIdentifier(freshName("n"))
    val m = NatIdentifier(freshName("m"))
    val k = NatIdentifier(freshName("k"))

    val rowAC = Identifier(freshName("vecAC"), ExpType(PairType(ArrayType(m, f32),ArrayType(k, f32)), read))
    val columnBFC = Identifier(freshName("columnBFC"), ExpType(PairType(ArrayType(m, f32),f32), read))
    val matA = Identifier(freshName("matA"), ExpType(ArrayType(n, ArrayType(m, f32)), read))
    val matB = Identifier(freshName("matB"), ExpType(ArrayType(m, ArrayType(k, f32)), read))
    val matC = Identifier(freshName("matC"), ExpType(ArrayType(n, ArrayType(k, f32)), read))

    val mul = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, Fst(f32, f32, x), Snd(f32, f32, x)))
    val add = Lambda[ExpType, FunType[ExpType, ExpType]](y, Lambda[ExpType, ExpType](z, BinOp(Operators.Binary.ADD, y, z)))

    val gemm = DepLambda[NatKind](n)(DepLambda[NatKind](m)(DepLambda[NatKind](k)(
      Lambda[ExpType, FunType[ExpType, FunType[ExpType, ExpType]]](matC,
        Lambda[ExpType, FunType[ExpType, ExpType]](matA,
          Lambda[ExpType, ExpType](matB,
            MapSeq(n, PairType(ArrayType(m, f32), ArrayType(k, f32)), ArrayType(k, f32),
              Lambda[ExpType, ExpType] (rowAC,
                MapSeq(k, PairType(ArrayType(m, f32), f32), f32,
                  Lambda[ExpType, ExpType] (columnBFC,
                    BinOp(Operators.Binary.ADD,
                      ReduceSeq(m, f32, f32, add, Literal(FloatData(0.0f)),
                        MapSeq(m, PairType(f32, f32), f32, mul,
                          Zip(m, f32, f32,
                            Fst(ArrayType(m,f32), ArrayType(k, f32), rowAC),
                            Fst(ArrayType(m,f32), f32, columnBFC)))),
                      Snd(ArrayType(m,f32), f32, columnBFC))),
                  Zip(k, ArrayType(m, f32), f32,
                    Transpose(m, k, f32, matB), Snd(ArrayType(m,f32), ArrayType(k, f32), rowAC)))),
              Zip(n, ArrayType(m, f32), ArrayType(k, f32), matA, matC))))))))

    println(ProgramGenerator.makeCode(gemm, "gemm").code)
  }

  test("matrixMult OpenCl") {


    // def dotproduct = fun((v,w) => reduce(+, 0) <<: map(*) <<: zip(v,w)) // rise
    // def matrixmultiplication = fun((v,w)) => map(map(dotproduct(v,transpose(w)))  => transpose(w)

    //def add(a: f32, b: f32):f32 = a + b

    val x = Identifier(freshName("x"), ExpType(PairType(f32, f32), read))
    val y = Identifier(freshName("y"), ExpType(f32, read))
    val z = Identifier(freshName("z"), ExpType(f32, read))

    val n = NatIdentifier(freshName("n"))
    val m = NatIdentifier(freshName("m"))
    val r = NatIdentifier(freshName("r"))
    val columnB = Identifier(freshName("columnB"), ExpType(ArrayType(m, f32), read))
    val rowA = Identifier(freshName("rowA"), ExpType(ArrayType(m, f32), read))
    val matrixA = Identifier(freshName("MatrixA"), ExpType(ArrayType(n, ArrayType(m,f32)),read))
    val matrixB = Identifier(freshName("MatrixB"), ExpType(ArrayType(m, ArrayType(r,f32)),read))

    val mul = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, Fst(f32, f32, x), Snd(f32, f32, x)))
    val add = Lambda[ExpType, FunType[ExpType, ExpType]](y, Lambda[ExpType, ExpType](z, BinOp(Operators.Binary.ADD, y, z)))

    val dotproduct = Lambda[ExpType, ExpType](rowA,
      OpenCLReduceSeq(m, shine.DPIA.Types.AddressSpace.Global, f32, f32, add, Literal(FloatData(0.0f)),
        To(shine.DPIA.Types.AddressSpace.Global, ArrayType(m, f32),
          MapSeq(m, PairType(f32, f32), f32, mul,
            Zip(m, f32, f32, rowA, columnB))), false))

    val matrixMult = DepLambda[NatKind](r)(DepLambda[NatKind](m)(DepLambda[NatKind](n)(
      Lambda[ExpType, FunType[ExpType, ExpType]](
        matrixA, Lambda[ExpType, ExpType](matrixB,
          MapThreads('x')(n, ArrayType(m,f32), ArrayType(r,f32),
            Lambda[ExpType, ExpType](columnB,
              MapThreads('y')(r, ArrayType(m,f32), f32,
                dotproduct,
                Transpose(m, r, f32,matrixB))),
            matrixA
          ))
      ))))

    val kernel = shine.cuda.KernelGenerator.apply().makeCode(matrixMult, "matrixMult")
    println("CODE:")
    println(kernel.code)
    val scalaFun = kernel.as[ScalaFunction`(`Int `,` Int `,` Int `,` scala.Array[scala.Array[Int]]`,` scala.Array[scala.Array[Int]]`)=>`scala.Array[scala.Array[Int]]].withSizes(LocalSize(1), GlobalSize(1))

    val vecAArray = scala.Array(scala.Array(1,1), scala.Array(1,1))
    val vecBArray = scala.Array(scala.Array(1,1), scala.Array(1,1))

    val (result, time) = scalaFun(2 `,` 2 `,` 2 `,` vecAArray `,` vecBArray)
    println(time)

    println(result(0)(0))
  }
}
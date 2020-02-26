package shine.examples

import shine.C.ProgramGenerator
import shine.DPIA.Phrases._
import shine.DPIA.Types._
import shine.DPIA.FunctionalPrimitives.{Fst, MapSeq, ReduceSeq, Snd, Transpose, Zip}
import shine.DPIA.Semantics.OperationalSemantics.FloatData
import shine.DPIA._
import shine.OpenCL.FunctionalPrimitives.{MapGlobal, OpenCLReduceSeq, To}
import shine.OpenCL._
import shine.cuda.primitives.functional.MapThreads
import shine.test_util
import util.SyntaxChecker

class matrixmultiplication extends test_util.Tests {

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

  // def dotproduct = fun((v,w) => reduce(+, 0) <<: map(*) <<: zip(v,w)) // rise

  val dotproduct = Lambda[ExpType, ExpType](columnB,
    ReduceSeq(m, f32, f32, add, Literal(FloatData(0f)),
      MapSeq(m, PairType(f32, f32), f32, mul,
        Zip(m, f32, f32, rowA, columnB))))

  val dotproductCL = Lambda[ExpType, ExpType](columnB,
    OpenCLReduceSeq(m, shine.OpenCL.AddressSpace.Global, f32, f32, add, Literal(FloatData(0f)),
      To(shine.OpenCL.AddressSpace.Global, ArrayType(m, f32),
        MapSeq(m, PairType(f32, f32), f32, mul,
          Zip(m, f32, f32, rowA, columnB))),
      false))

  val matrixATest = scala.Array(scala.Array(1f, 2f), scala.Array(3f, 4f))
  val matrixBTest = scala.Array(scala.Array(1f, 2f), scala.Array(3f, 4f))
  val resultTest = matrixMult(matrixATest, matrixBTest)

  test("matrixMult C") {
    // def matrixmultiplication = fun((v,w)) => map(map(dotproduct(v,transpose(w)))  => transpose(w)

    val matrixMult = DepLambda[NatKind](r)(DepLambda[NatKind](m)(DepLambda[NatKind](n)(
      Lambda[ExpType, FunType[ExpType, ExpType]](matrixA,
        Lambda[ExpType, ExpType](matrixB,
          MapSeq(n, ArrayType(m, f32), ArrayType(r, f32),
            Lambda[ExpType, ExpType](rowA,
              MapSeq(r, ArrayType(m, f32), f32,
                dotproduct,
                Transpose(m, r, f32, matrixB))),
            matrixA))))))

    println(ProgramGenerator.makeCode(matrixMult, "matrixMult").code)
  }

  testCL("matrixMult OpenCL") {
    val matrixMult = DepLambda[NatKind](r)(DepLambda[NatKind](m)(DepLambda[NatKind](n)(
      Lambda[ExpType, FunType[ExpType, ExpType]](matrixA,
        Lambda[ExpType, ExpType](matrixB,
          MapGlobal(0)(n, ArrayType(m, f32), ArrayType(r, f32),
            Lambda[ExpType, ExpType](rowA,
              MapGlobal(1)(r, ArrayType(m, f32), f32,
                dotproductCL,
                Transpose(m, r, f32, matrixB))),
            matrixA))))))

    val kernel = shine.OpenCL.KernelGenerator.apply().makeCode(matrixMult, "matrixMult")
    SyntaxChecker.checkOpenCL(kernel.code)

    checkMatrixMultKernel(kernel)
  }

  testCU("matrixMult CUDA") {
    val matrixMult = DepLambda[NatKind](r)(DepLambda[NatKind](m)(DepLambda[NatKind](n)(
      Lambda[ExpType, FunType[ExpType, ExpType]](matrixA,
        Lambda[ExpType, ExpType](matrixB,
          MapThreads('x')(n, ArrayType(m, f32), ArrayType(r, f32),
            Lambda[ExpType, ExpType](rowA,
              MapThreads('y')(r, ArrayType(m, f32), f32,
                dotproductCL,
                Transpose(m, r, f32, matrixB))),
            matrixA))))))

    val kernel = shine.cuda.KernelGenerator.apply().makeCode(matrixMult, "matrixMult")

    checkMatrixMultKernel(kernel)
  }

  private def checkMatrixMultKernel(kernel: util.KernelNoSizes): Unit ={
    val scalaFun = kernel.as[ScalaFunction`(`Int `,` Int `,` Int `,` scala.Array[scala.Array[Float]] `,` scala.Array[scala.Array[Float]] `)=>` scala.Array[Float]].withSizes(LocalSize(1), GlobalSize(1))

    val (result, _) = scalaFun(matrixATest.length `,` matrixBTest.length `,` matrixBTest.transpose.length `,` matrixATest `,` matrixBTest)

    val resultMatrix = result.sliding(matrixBTest.length, matrixBTest.length).toArray

    if (!similar(resultMatrix, resultTest)){
      println("Expected: ")
      println(resultTest.deep.mkString("\n"))
      println("Result: ")
      println(resultMatrix.deep.mkString("\n"))

      println("KernelCode:")
      println(kernel.code)

      throw new RuntimeException("false result")
    }
  }

  /**
    * Multiply matrixA with matrixB using scala.
    * @param matrixA first matrix
    * @param matrixB second matrix
    * @return product of matrixA and matrixB
    */
  private def matrixMult(matrixA: scala.Array[scala.Array[Float]], matrixB: scala.Array[scala.Array[Float]]) : scala.Array[scala.Array[Float]] = {
    assert(matrixA.transpose.length == matrixB.length)

    matrixA.map(rowA =>
      matrixB.transpose.map(columnB =>
        (rowA zip columnB
          map Function.tupled(_ * _)).sum))
  }
}
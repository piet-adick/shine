package shine.examples

import shine.C.ProgramGenerator
import shine.DPIA.FunctionalPrimitives.{Fst, MapSeq, ReduceSeq, Snd, Transpose, Zip}
import shine.DPIA.Phrases.{BinOp, DepLambda, Identifier, Lambda, Literal, Operators}
import shine.DPIA.Semantics.OperationalSemantics.FloatData
import shine.DPIA.Types.{ArrayType, ExpType, FunType, NatKind, PairType, f32, read}
import shine.DPIA.{NatIdentifier, freshName}
import shine.OpenCL.FunctionalPrimitives.{MapGlobal, OpenCLReduceSeq, To}
import shine.OpenCL._
import shine.cuda.primitives.functional.MapThreads
import shine.test_util
import util.{KernelNoSizes, SyntaxChecker}

//GEMM = general matrix multiply
//With A,B,C matrices
//This calculate result = alpha * A*B + beta * C
class gemm extends test_util.Tests {

  val x = Identifier(freshName("x"), ExpType(PairType(f32, f32), read))
  val y = Identifier(freshName("y"), ExpType(f32, read))
  val z = Identifier(freshName("z"), ExpType(f32, read))

  val n = NatIdentifier(freshName("n"))
  val m = NatIdentifier(freshName("m"))
  val k = NatIdentifier(freshName("k"))

  val matA = Identifier(freshName("matA"), ExpType(ArrayType(n, ArrayType(m, f32)), read))
  val matB = Identifier(freshName("matB"), ExpType(ArrayType(m, ArrayType(k, f32)), read))
  val matC = Identifier(freshName("matC"), ExpType(ArrayType(n, ArrayType(k, f32)), read))
  val alpha = Identifier(freshName("alpha"), ExpType(f32, read))
  val beta = Identifier(freshName("beta"), ExpType(f32, read))
  //pair with row of MatrixA and row of matrix C
  val rowAC = Identifier(freshName("rowAC"), ExpType(PairType(ArrayType(m, f32),ArrayType(k, f32)), read))
  //piar with column of MatrixB and single float of matrix C
  val columnBElementC = Identifier(freshName("columnBElementC"), ExpType(PairType(ArrayType(m, f32), f32), read))

  val mul = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, Fst(f32, f32, x), Snd(f32, f32, x)))
  val add = Lambda[ExpType, FunType[ExpType, ExpType]](y, Lambda[ExpType, ExpType](z, BinOp(Operators.Binary.ADD, y, z)))

  // def dotproduct = fun((v,w) => reduce(+, 0) <<: map(*) <<: zip(v,w)) // rise

  val dotproduct = ReduceSeq(m, f32, f32, add, Literal(FloatData(0.0f)),
    MapSeq(m, PairType(f32, f32), f32, mul,
      Zip(m, f32, f32,
        Fst(ArrayType(m,f32), ArrayType(k, f32), rowAC),
        Fst(ArrayType(m,f32), f32, columnBElementC))))

  val dotproductCL = OpenCLReduceSeq(m, shine.OpenCL.AddressSpace.Global, f32, f32, add, Literal(FloatData(0f)),
      To(shine.OpenCL.AddressSpace.Global, ArrayType(m, f32),
        MapSeq(m, PairType(f32, f32), f32, mul,
          Zip(m, f32, f32,
            Fst(ArrayType(m,f32), ArrayType(k, f32), rowAC),
            Fst(ArrayType(m,f32), f32, columnBElementC)))),
      false)

  val matrixATest = scala.Array(scala.Array(1f, 2f), scala.Array(3f, 4f))
  val matrixBTest = scala.Array(scala.Array(1f, 2f), scala.Array(3f, 4f))
  val matrixCTest = scala.Array(scala.Array(39f, 40f), scala.Array(41f, 42f))
  val alphaTest = 1f;
  val betaTest = 2f;
  val resultTest = gemm(matrixATest, matrixBTest, matrixCTest, alphaTest, betaTest)

  test("GEMM C") {
    val gemm = DepLambda[NatKind](n)(DepLambda[NatKind](m)(DepLambda[NatKind](k)(
      //take matrixA as parameter
      Lambda[ExpType, FunType[ExpType, FunType[ExpType, FunType[ExpType, FunType[ExpType, ExpType]]]]](matA,
        //take matrixB as parameter
        Lambda[ExpType, FunType[ExpType, FunType[ExpType, FunType[ExpType, ExpType]]]](matB,
          //take matrixC as parameter
          Lambda[ExpType, FunType[ExpType, FunType[ExpType, ExpType]]](matC,
            //take alpha as parameter
            Lambda[ExpType, FunType[ExpType, ExpType]](alpha,
              //take beta as parameter
              Lambda[ExpType, ExpType](beta,
                //map over pair of matrixA and matrixC
                MapSeq(n, PairType(ArrayType(m, f32), ArrayType(k, f32)), ArrayType(k, f32),
                  //elements are pairs of row of matrixA and row of matrixC
                  Lambda[ExpType, ExpType] (rowAC,
                    //map over transposed matrixB and row of matrixC
                    MapSeq(k, PairType(ArrayType(m, f32), f32), f32,
                      //elements are pairs of columns of matrixB and elements of matrixC
                      Lambda[ExpType, ExpType] (columnBElementC,
                        BinOp(Operators.Binary.ADD,
                          //alpha * dotproduct of rowA and columnB
                          BinOp(Operators.Binary.MUL,
                            alpha,
                            //dotproduct of rowA and columnB
                            dotproduct),
                          //beta * element of matrixC
                          BinOp(Operators.Binary.MUL,
                            beta,
                            //element of matrixC
                            Snd(ArrayType(m,f32), f32, columnBElementC)))),
                      //zip transposed matrixB and row of matrixC
                      Zip(k, ArrayType(m, f32), f32,
                        Transpose(m, k, f32, matB),
                        Snd(ArrayType(m,f32), ArrayType(k, f32), rowAC)))),
                  //zip matrixA and matrixB
                  Zip(n, ArrayType(m, f32), ArrayType(k, f32), matA, matC))))))))))

    val kernel = KernelNoSizes(ProgramGenerator.makeCode(gemm, "gemm"))

    println("Gemm C-Code:")
    println(kernel.code)

    checkGEMMKernel(kernel)
  }

  testCL("GEMM OpenCL") {
    val gemm = DepLambda[NatKind](n)(DepLambda[NatKind](m)(DepLambda[NatKind](k)(
      Lambda[ExpType, FunType[ExpType, FunType[ExpType, FunType[ExpType, FunType[ExpType, ExpType]]]]](matA,
        Lambda[ExpType, FunType[ExpType, FunType[ExpType, FunType[ExpType, ExpType]]]](matB,
          Lambda[ExpType, FunType[ExpType, FunType[ExpType, ExpType]]](matC,
            Lambda[ExpType, FunType[ExpType, ExpType]](alpha,
              Lambda[ExpType, ExpType](beta,
                MapGlobal(0)(n, PairType(ArrayType(m, f32), ArrayType(k, f32)), ArrayType(k, f32),
                  Lambda[ExpType, ExpType] (rowAC,
                    MapGlobal(0)(k, PairType(ArrayType(m, f32), f32), f32,
                      Lambda[ExpType, ExpType] (columnBElementC,
                        BinOp(Operators.Binary.ADD,
                          BinOp(Operators.Binary.MUL,
                            alpha,
                            dotproductCL),
                          BinOp(Operators.Binary.MUL,
                            beta,
                            Snd(ArrayType(m,f32), f32, columnBElementC)))),
                      Zip(k, ArrayType(m, f32), f32,
                        Transpose(m, k, f32, matB),
                        Snd(ArrayType(m,f32), ArrayType(k, f32), rowAC)))),
                  Zip(n, ArrayType(m, f32), ArrayType(k, f32), matA, matC))))))))))

    val kernel = shine.OpenCL.KernelGenerator.apply().makeCode(gemm, "gemm")
    SyntaxChecker.checkOpenCL(kernel.code)

    checkGEMMKernel(kernel)
  }

  testCU("GEMM CUDA") {
    val gemm = DepLambda[NatKind](n)(DepLambda[NatKind](m)(DepLambda[NatKind](k)(
      Lambda[ExpType, FunType[ExpType, FunType[ExpType, FunType[ExpType, FunType[ExpType, ExpType]]]]](matA,
        Lambda[ExpType, FunType[ExpType, FunType[ExpType, FunType[ExpType, ExpType]]]](matB,
          Lambda[ExpType, FunType[ExpType, FunType[ExpType, ExpType]]](matC,
            Lambda[ExpType, FunType[ExpType, ExpType]](alpha,
              Lambda[ExpType, ExpType](beta,
                MapThreads('x')(n, PairType(ArrayType(m, f32), ArrayType(k, f32)), ArrayType(k, f32),
                  Lambda[ExpType, ExpType] (rowAC,
                    MapThreads('y')(k, PairType(ArrayType(m, f32), f32), f32,
                      Lambda[ExpType, ExpType] (columnBElementC,
                        BinOp(Operators.Binary.ADD,
                          BinOp(Operators.Binary.MUL,
                            alpha,
                            dotproductCL),
                          BinOp(Operators.Binary.MUL,
                            beta,
                            Snd(ArrayType(m,f32), f32, columnBElementC)))),
                      Zip(k, ArrayType(m, f32), f32,
                        Transpose(m, k, f32, matB),
                        Snd(ArrayType(m,f32), ArrayType(k, f32), rowAC)))),
                  Zip(n, ArrayType(m, f32), ArrayType(k, f32), matA, matC))))))))))

    val kernel = shine.cuda.KernelGenerator.apply().makeCode(gemm, "gemm")

    checkGEMMKernel(kernel)
  }

  private def checkGEMMKernel(kernel: util.KernelNoSizes): Unit ={
    val scalaFun = kernel.as[ScalaFunction`(`Int `,` Int `,` Int `,` scala.Array[scala.Array[Float]] `,` scala.Array[scala.Array[Float]] `,` scala.Array[scala.Array[Float]] `,` Float `,` Float `)=>` scala.Array[Float]].withSizes(LocalSize(1), GlobalSize(1))

    val (result, _) = scalaFun(matrixATest.length `,` matrixBTest.length `,` matrixBTest.transpose.length `,` matrixATest `,` matrixBTest `,` matrixCTest `,` alphaTest `,` betaTest)

    val resultTestArray = resultTest.flatMap(_.toList)

    if (!(result sameElements resultTestArray)){
      val resultMatrix = result.sliding(matrixBTest.length, matrixBTest.length).toArray

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
    * Calculate gemm (matrixA*matrixB + matrixC).
    * @param matrixA first matrix of product
    * @param matrixB second matrix of product
    * @param matrixC second matrix of product
    * @return product of matrixA and matrixB
    */
  private def gemm(matrixA: scala.Array[scala.Array[Float]], matrixB: scala.Array[scala.Array[Float]], matrixC: scala.Array[scala.Array[Float]], alpha: Float, beta: Float) : scala.Array[scala.Array[Float]] = {
    val nDim = matrixA.length
    val mDim = matrixB.length
    val kDim = matrixB.transpose.length

    assert(matrixA.transpose.length == mDim)
    assert(matrixC.length == nDim)
    assert(matrixC.transpose.length == kDim)

    val matrixProduct = matrixA.map(rowA =>
      matrixB.transpose.map(columnB =>
        (rowA zip columnB
          map Function.tupled(_ * _)).sum))

    for (y <- 0 until nDim; x <- 0 until kDim) {
      matrixProduct(y)(x) = alpha * matrixProduct(y)(x) + beta * matrixC(y)(x)
    }

    matrixProduct
  }
}

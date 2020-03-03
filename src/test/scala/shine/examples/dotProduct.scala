package shine.examples

import shine.C.ProgramGenerator
import shine.DPIA.Phrases._
import shine.DPIA.Types._
import shine.DPIA.FunctionalPrimitives._
import shine.DPIA.Semantics.OperationalSemantics.FloatData
import shine.DPIA._
import shine.OpenCL.FunctionalPrimitives.{OpenCLReduceSeq, To}
import shine.OpenCL._
import shine.test_util
import util.SyntaxChecker

class dotProduct extends test_util.Tests {

  val x = Identifier(freshName("x"), ExpType(PairType(f32, f32), read))
  val y = Identifier(freshName("y"), ExpType(f32, read))
  val z = Identifier(freshName("z"), ExpType(f32, read))

  val n = NatIdentifier(freshName("n"))
  val vecA = Identifier(freshName("vecA"), ExpType(ArrayType(n, f32), read))
  val vecB = Identifier(freshName("vecB"), ExpType(ArrayType(n, f32), read))

  val mul = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, Fst(f32, f32, x), Snd(f32, f32, x)))
  val add = Lambda[ExpType, FunType[ExpType, ExpType]](y, Lambda[ExpType, ExpType](z, BinOp(Operators.Binary.ADD, y, z)))

  val vecATest = scala.Array(1f, 2f, 3f, 4f)
  val vecBTest = scala.Array(1f, 2f, 3f, 4f)
  val resultTest = dotproduct(vecATest, vecBTest)

  test("dotproduct C") {
    // def dotproduct = fun((v,w) => reduce(+, 0) <<: map(*) <<: zip(v,w)) // rise

    val dot = DepLambda[NatKind](n)(
      Lambda[ExpType, FunType[ExpType, ExpType]](vecA,
        Lambda[ExpType, ExpType](vecB,
          ReduceSeq(n, f32, f32, add, Literal(FloatData(0f)),
            MapSeq(n, PairType(f32, f32), f32, mul,
              Zip(n, f32, f32, vecA, vecB))))))

    println(ProgramGenerator.makeCode(dot, "dot-product").code)
  }

  testCL("dotproduct OpenCL") {
    val dot = DepLambda[NatKind](n)(
      Lambda[ExpType, FunType[ExpType, ExpType]](vecA,
        Lambda[ExpType, ExpType](vecB,
          OpenCLReduceSeq(n, shine.DPIA.Types.AddressSpace.Global, f32, f32, add, Literal(FloatData(0f)),
            To(shine.DPIA.Types.AddressSpace.Global, ArrayType(n, f32),
              MapSeq(n, PairType(f32, f32), f32, mul,
                Zip(n, f32, f32, vecA, vecB))),
            false))))

    val kernel = shine.OpenCL.KernelGenerator.apply().makeCode(dot, "dotProduct")
    SyntaxChecker.checkOpenCL(kernel.code)

    println("KernelCode:")
    println(kernel.code)

    checkDotKernel(kernel)
  }

  testCU("dotproduct CUDA") {
    val dot = DepLambda[NatKind](n)(
      Lambda[ExpType, FunType[ExpType, ExpType]](vecA,
        Lambda[ExpType, ExpType](vecB,
          OpenCLReduceSeq(n, shine.DPIA.Types.AddressSpace.Global, f32, f32, add, Literal(FloatData(0f)),
            To(shine.DPIA.Types.AddressSpace.Global, ArrayType(n, f32),
              MapSeq(n, PairType(f32, f32), f32, mul,
                Zip(n, f32, f32, vecA, vecB))),
            false))))

    val kernel = shine.cuda.KernelGenerator.apply().makeCode(dot, "dotProduct")

    checkDotKernel(kernel)
  }

  private def checkDotKernel(kernel: util.KernelNoSizes): Unit ={
    val scalaFun = kernel.as[ScalaFunction`(`Int `,` scala.Array[Float]`,` scala.Array[Float]`)=>`scala.Array[Float]].withSizes(LocalSize(1), GlobalSize(1))

    val (result, _) = scalaFun(vecATest.length `,` vecATest `,` vecBTest)

    if (!similar(result(0), resultTest)){
      print("Expected: ")
      println(resultTest)
      print("Result: ")
      println(result(0))

      println("KernelCode:")
      println(kernel.code)

      throw new RuntimeException("false result")
    }
  }

  /**
    * Calculate dot product of vecA and vecB using scala.
    * @param vecA first vector
    * @param vecB second vector
    * @return dot product of vecA and vecB
    */
  private def dotproduct(vecA: scala.Array[Float], vecB: scala.Array[Float]) : Float = {
    assert(vecA.length == vecB.length)

    (vecA zip vecB).map{Function.tupled(_ * _)}.sum
  }
}
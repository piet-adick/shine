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
import util.{KernelNoSizes}

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

  testCU("dotproduct C") {
    // def dotproduct = fun((v,w) => reduce(+, 0) <<: map(*) <<: zip(v,w)) // rise

    val dot = DepLambda[NatKind](n)(
      Lambda[ExpType, FunType[ExpType, ExpType]](vecA,
        Lambda[ExpType, ExpType](vecB,
          ReduceSeq(n, f32, f32, add, Literal(FloatData(0f)),
            MapSeq(n, PairType(f32, f32), f32, mul,
              Zip(n, f32, f32, vecA, vecB))))))

    val kernel = KernelNoSizes(ProgramGenerator.makeCode(dot, "dotProduct"))

    println("dotProduct C-Code:")
    println(kernel.code)

    println("dotProduct C DataSize")
    checkDotKernel(kernel)
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

    println("dotProduct OpenCL: DataSize")
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

    println("dotProduct CUDA: DataSize")
    checkDotKernel(kernel)
  }

  val KB = 1024l
  val MB = KB*KB

  val dataSizes = scala.Array(
  1 * KB,
  4 * KB,
  16 * KB,
  64 * KB)
//  256 * KB,
//  1 * MB,
  //4 * MB)
 // 16 * MB,
 // 64 * MB,
 // 256 * MB)

  private def checkDotKernel(kernel: util.KernelNoSizes): Unit ={

    val scalaFun = kernel.as[ScalaFunction`(`Int `,` scala.Array[Float]`,` scala.Array[Float]`)=>`scala.Array[Float]].withSizes(LocalSize(1), GlobalSize(1))

    //Benchmark
    for (dataSize <- dataSizes){
      val n = (dataSize/4).asInstanceOf[Int]

      val x: scala.Array[Float] = new scala.Array[Float](n)
      val y: scala.Array[Float] = new scala.Array[Float](n)

      for (i <- 0 until n) {
        x(i) = i
        y(i) = 2 * i
      }

      val (_, runtime) = scalaFun(n `,` x `,` y)

      print("DataSize: " + Print.humanReadableByteCountBin(dataSize) + " ")
      println("execution-time: " + runtime)
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
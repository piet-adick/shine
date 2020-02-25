package shine.examples

import shine.C.ProgramGenerator
import shine.DPIA.Phrases._
import shine.DPIA.Types._
import shine.DPIA.FunctionalPrimitives._
import shine.DPIA.Semantics.OperationalSemantics.{IntData, FloatData}
import shine.DPIA._
import shine.OpenCL.FunctionalPrimitives.{OpenCLReduceSeq, To}
import shine.OpenCL._
import shine.test_util

//TODO refactor
class dotProduct extends test_util.TestsWithYACX {

  test("dot-product") {
    /* javascript:
    const zip = (arr, ...arrs) => {
      return arr.map((val, i) => arrs.reduce((a, arr) => [...a, arr[i]], [val]));
    }
    const dot = (a, b) => zip(a, b).map(x => x[0] * x[1]).reduce((result, adder) => result + adder, 0)
    dot([1,2,3],[4,5,6])
    => 32
     */

    /*
    [1,2,3],[4,5,6] // zippen
    [[1,4],[2,5],[3,6]] // map mit pair mul
    [4,10,18] // reduce sum
    [32]
     */

    // def dotproduct = fun((v,w) => reduce(+, 0) <<: map(*) <<: zip(v,w)) // rise


    //def add(a: f32, b: f32):f32 = a + b

    val x = Identifier(freshName("x"), ExpType(PairType(f32, f32), read))
    val y = Identifier(freshName("y"), ExpType(f32, read))
    val z = Identifier(freshName("z"), ExpType(f32, read))

    val n = NatIdentifier(freshName("n"))
    val vecA = Identifier(freshName("vecA"), ExpType(ArrayType(n, f32), read))
    val vecB = Identifier(freshName("vecB"), ExpType(ArrayType(n, f32), read))


    val mul = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, Fst(f32, f32, x), Snd(f32, f32, x)))
    val add = Lambda[ExpType, FunType[ExpType, ExpType]](y, Lambda[ExpType, ExpType](z, BinOp(Operators.Binary.ADD, y, z)))

    val dot = DepLambda[NatKind](n)(Lambda[ExpType, FunType[ExpType, ExpType]](vecA, Lambda[ExpType, ExpType](vecB,
      ReduceSeq(n, f32, f32, add, Literal(FloatData(0.0f)),
        MapSeq(n, PairType(f32, f32), f32, mul,
          Zip(n, f32, f32, vecA, vecB))))))

    println(ProgramGenerator.makeCode(dot, "dot-product").code)

    /*
    println(add(1,2))
    add(8.0f,2.0f) shouldBe 10
    */
  }

//  test("dotproduct OpenCL") {
//    val x = Identifier(freshName("x"), ExpType(PairType(int, int), read))
//    val y = Identifier(freshName("y"), ExpType(int, read))
//    val z = Identifier(freshName("z"), ExpType(int, read))
//
//    val n = NatIdentifier(freshName("n"))
//    val vecA = Identifier(freshName("vecA"), ExpType(ArrayType(n, int), read))
//    val vecB = Identifier(freshName("vecB"), ExpType(ArrayType(n, int), read))
//
//    val mul = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, Fst(int, int, x), Snd(int, int, x)))
//    val add = Lambda[ExpType, FunType[ExpType, ExpType]](y, Lambda[ExpType, ExpType](z, BinOp(Operators.Binary.ADD, y, z)))
//
//    val dot = DepLambda[NatKind](n)(Lambda[ExpType, FunType[ExpType, ExpType]](vecA, Lambda[ExpType, ExpType](vecB,
//      //Wird entsprechendes OpenCL Pattern verwendet?
//      OpenCLReduceSeq(n, shine.DPIA.Types.AddressSpace.Global, int, int, add, Literal(IntData(0)),
//        //Falls ein new without address space fehler auftaucht: fehlt ein To um ein MapSeq (outermost häufig wichtigste)?
//        To(shine.DPIA.Types.AddressSpace.Global, ArrayType(n, int), MapSeq(n, PairType(int, int), int, mul,
//          Zip(n, int, int, vecA, vecB))), false))))
//
//    val kernel = KernelGenerator.apply().makeCode(dot, "dotProduct")
//    println("CODE:")
//    println(kernel.code)
//    SyntaxChecker.checkOpenCL(kernel.code)
//    val scalaFun = kernel.as[ScalaFunction`(`Int `,` scala.Array[Int]`,` scala.Array[Int]`)=>`scala.Array[Int]].withSizes(LocalSize(1), GlobalSize(1))
//
//    val vecAArray = scala.Array.fill(4)(1)
//    val vecBArray = scala.Array.fill(4)(1)
//
//    val (result, time) = scalaFun(4 `,` vecAArray `,` vecBArray)
//    println(time)
//
//    println(result(0))
//  }

test("dotproduct CUDA") {
  val x = Identifier(freshName("x"), ExpType(PairType(int, int), read))
  val y = Identifier(freshName("y"), ExpType(int, read))
  val z = Identifier(freshName("z"), ExpType(int, read))

  val n = NatIdentifier(freshName("n"))
  val vecA = Identifier(freshName("vecA"), ExpType(ArrayType(n, int), read))
  val vecB = Identifier(freshName("vecB"), ExpType(ArrayType(n, int), read))

  val mul = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, Fst(int, int, x), Snd(int, int, x)))
  val add = Lambda[ExpType, FunType[ExpType, ExpType]](y, Lambda[ExpType, ExpType](z, BinOp(Operators.Binary.ADD, y, z)))

  val dot = DepLambda[NatKind](n)(Lambda[ExpType, FunType[ExpType, ExpType]](vecA, Lambda[ExpType, ExpType](vecB,
  //Wird entsprechendes OpenCL Pattern verwendet?
  OpenCLReduceSeq(n, shine.DPIA.Types.AddressSpace.Global, int, int, add, Literal(IntData(0)),
  //Falls ein new without address space fehler auftaucht: fehlt ein To um ein MapSeq (outermost häufig wichtigste)?
  To(shine.DPIA.Types.AddressSpace.Global, ArrayType(n, int), MapSeq(n, PairType(int, int), int, mul,
  Zip(n, int, int, vecA, vecB))), false))))

  val kernel = shine.cuda.KernelGenerator.apply().makeCode(dot, "dotProduct")
  println("CODE:")
  println(kernel.code)
  val scalaFun = kernel.as[ScalaFunction`(`Int `,` scala.Array[Int]`,` scala.Array[Int]`)=>`scala.Array[Int]].withSizes(LocalSize(1), GlobalSize(1))

  val vecAArray = scala.Array.fill(4)(1)
  val vecBArray = scala.Array.fill(4)(1)

  val (result, time) = scalaFun(4 `,` vecAArray `,` vecBArray)
  println(time)

  println(result(0))
}
}
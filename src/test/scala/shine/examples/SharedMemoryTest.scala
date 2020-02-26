package shine.examples

import rise.core.types.NatToNatIdentifier
import shine.DPIA.FunctionalPrimitives.{Join, Split}
import shine.DPIA.Phrases.Phrase.Internal
import shine.DPIA.Phrases.{DepLambda, Identifier, Lambda, Operators}
import shine.DPIA.Types.{AddressSpace, ArrayType, ExpType, NatKind, int, read}
import shine.DPIA.{NatIdentifier, Types, freshName}
import shine.OpenCL.FunctionalPrimitives.To
import shine.OpenCL._
import shine.cuda.primitives.functional.{MapBlock, MapGrid}
import shine.test_util

class SharedMemoryTest extends test_util.Tests {
  val chunkSize = NatIdentifier(freshName("chunkSize"))
  val n = NatIdentifier(freshName("n"))
  val array = Identifier(freshName("array"), ExpType(ArrayType(n, int), read))
  val chunk = Identifier(freshName("chunk"), ExpType(ArrayType(chunkSize, int), read))
  val x = Identifier(freshName("x"), ExpType(int, read))

  val chunkSizeTest = 2
  val arrayTest = Array(1, 2, 3, 4, 5, 6, 7, 8)
  val resultArray = Array(1, 2, 3, 4, 5, 6, 7, 8)




  testCU("SharedMemory-test CUDA"){
    val copy = Lambda[ExpType, ExpType](chunk,
      MapBlock('x')(chunkSize, int, int,
        Lambda[ExpType, ExpType](x, x),
        To(AddressSpace.Global, ArrayType(chunkSize, int),
          MapBlock('x')(chunkSize, int, int,
            Lambda[ExpType, ExpType](x, x), chunk))
    ))
    //n/chunksize
    //val divid = Types.NatT(Internal.binOpToNat(Operators.Binary.DIV, n,chunkSize))

    val test = DepLambda[NatKind](chunkSize)(DepLambda[NatKind](n)(
      Lambda[ExpType, ExpType](array,
        Join(n /^ chunkSize, chunkSize, read, int,
          MapGrid('x')(n /^ chunkSize, ArrayType(chunkSize, int), ArrayType(chunkSize, int), copy,
            Split(chunkSize, n /^ chunkSize, read, int, array))
    ))))

    val kernel = shine.cuda.KernelGenerator.apply().makeCode(test, "test")

    println("CODE:")
    println(kernel.code)
    println("\n")

    check(kernel)
  }

  private def check(kernel: util.KernelNoSizes): Unit ={
    val scalaFun = kernel.as[ScalaFunction`(`Int`,` Int`,` scala.Array[Int]`)=>`scala.Array[Int]].withSizes(LocalSize(1), GlobalSize(1))

    val (result, _) = scalaFun(chunkSizeTest `,` arrayTest.length `,` arrayTest)

    println("Expected: ")
    println(resultArray.deep.mkString("\n"))
    println("Result: ")
    println(result.deep.mkString("\n"))
  }
}

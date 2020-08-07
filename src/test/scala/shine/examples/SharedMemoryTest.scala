package shine.examples

import shine.DPIA.FunctionalPrimitives.{Join, Split}
import shine.DPIA.Phrases.{BinOp, DepLambda, Identifier, Lambda, Operators}
import shine.DPIA.Types.{AddressSpace, ArrayType, ExpType, NatKind, int, read}
import shine.DPIA.{NatIdentifier, freshName}
import shine.OpenCL.FunctionalPrimitives.{MapLocal, MapWorkGroup, To}
import shine.OpenCL._
import shine.cuda.primitives.functional.{MapBlock, MapGlobal}
import shine.test_util

class SharedMemoryTest extends test_util.Tests {
  val chunkSize = NatIdentifier(freshName("chunkSize"))
  val n = NatIdentifier(freshName("n"))
  val chunk = Identifier(freshName("chunk"), ExpType(ArrayType(chunkSize, int), read))
  val array = Identifier(freshName("array"), ExpType(ArrayType(n, int), read))
  val x = Identifier(freshName("x"), ExpType(int, read))

  //Identity function
  val id = Lambda[ExpType, ExpType](x, x)
  //Square function
  val square = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, x, x))

  val chunkSizeTest = 2
  val arrayTest = Array(1, 2, 3, 4, 5, 6, 7, 8)
  val resultArray = square(arrayTest)

  testCL("LocalMemory-test OpenCL"){
    val squareShared = Lambda[ExpType, ExpType](chunk,
      MapLocal(0)(chunkSize, int, int, id,
        To(AddressSpace.Local, ArrayType(chunkSize, int),
          MapLocal(0)(chunkSize, int, int, square, chunk))
      ))

    val test = DepLambda[NatKind](chunkSize)(DepLambda[NatKind](n)(
      Lambda[ExpType, ExpType](array,
        Join(n /^ chunkSize, chunkSize, read, int,
          MapWorkGroup(0)(n /^ chunkSize, ArrayType(chunkSize, int), ArrayType(chunkSize, int), squareShared,
            Split(chunkSize, n /^ chunkSize, read, int, array))
        ))))

    val kernel = shine.OpenCL.KernelGenerator.apply().makeCode(test, "testLocal")

    checkSquareKernel(kernel)
  }

  testCU("SharedMemory-test CUDA"){
    val squareShared = Lambda[ExpType, ExpType](chunk,
      MapBlock('x')(chunkSize, int, int, id,
        //TODO: use shared memory
        To(AddressSpace.Global, ArrayType(chunkSize, int),
          MapBlock('x')(chunkSize, int, int, square, chunk))
    ))

    val test = DepLambda[NatKind](chunkSize)(DepLambda[NatKind](n)(
      Lambda[ExpType, ExpType](array,
        Join(n /^ chunkSize, chunkSize, read, int,
          MapGlobal('x')(n /^ chunkSize, ArrayType(chunkSize, int), ArrayType(chunkSize, int), squareShared,
            Split(chunkSize, n /^ chunkSize, read, int, array))
    ))))

    val kernel = shine.cuda.KernelGenerator.apply().makeCode(test, "testShared")

    checkSquareKernel(kernel)
  }

  private def checkSquareKernel(kernel: util.KernelNoSizes): Unit ={
    val scalaFun = kernel.as[ScalaFunction`(`Int`,` Int`,` scala.Array[Int]`)=>`scala.Array[Int]].withSizes(LocalSize(1), GlobalSize(1))

    val (result, _) = scalaFun(chunkSizeTest `,` arrayTest.length `,` arrayTest)

    if (!(resultArray sameElements result)){
      println("Expected: ")
      println(resultArray.deep.mkString(", "))
      println("Result: ")
      println(result.deep.mkString(", "))

      println("KernelCode:")
      println(kernel.code)

      throw new RuntimeException("false result")
    }
  }

  /**
    * Calculate square of int-Array.
    * @param array int-array
    * @return array with square of ints in array
    */
  private def square(array: scala.Array[Int]) : scala.Array[Int] = {
    assert(arrayTest.length % chunkSizeTest == 0)

    array.map(math.pow(_, 2).asInstanceOf[Int])
  }
}

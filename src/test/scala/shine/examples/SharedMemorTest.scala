package shine.examples

import shine.DPIA.FunctionalPrimitives.{Join, Split}
import shine.DPIA.Phrases.{DepLambda, Identifier, Lambda}
import shine.DPIA.Types.{AddressSpace, ArrayType, ExpType, NatKind, int, read}
import shine.DPIA.{NatIdentifier, freshName}
import shine.OpenCL.FunctionalPrimitives.To
import shine.OpenCL.{GlobalSize, LocalSize, ScalaFunction, `(`, `)=>`, `,`}
import shine.cuda.primitives.functional.MapGrid
import shine.test_util

class SharedMemorTest extends test_util.Tests {
  val chunkSize = NatIdentifier(freshName("chunkSize"))
  val n = NatIdentifier(freshName("n"))
  val array = Identifier(freshName("array"), ExpType(ArrayType(n, int), read))
  val chunk = Identifier(freshName("chunk"), ExpType(ArrayType(chunkSize, int), read))



  testCU("SharedMemory-test CUDA"){
    val copy = Lambda[ExpType, ExpType](chunk,
      To(AddressSpace.Global, ArrayType(chunkSize, int), //copy From
        //Fkt F
        To(AddressSpace.Global, ArrayType(chunkSize, int), chunk) //copy To
      )
    )

    val test = DepLambda[NatKind](chunkSize)(DepLambda[NatKind](n)(
      Lambda[ExpType, ExpType](array,
        Join(chunkSize, n/chunkSize, read, int,
          MapGrid(0)(n/chunkSize, ArrayType(chunkSize, int), ArrayType(chunkSize, int), copy,
            Split(chunkSize, n/chunkSize, read, int, array))
    ))))


  }

  private def check(kernel: util.KernelNoSizes): Unit ={
    val scalaFun = kernel.as[ScalaFunction`(`scala.Array[Int]`,` scala.Array[Int]`)=>`scala.Array[Int]].withSizes(LocalSize(1), GlobalSize(1))

    val (result, _) = scalaFun(matrixATest.length `,` matrixBTest.length `,` matrixBTest.transpose.length `,` matrixATest `,` matrixBTest)

    val resultMatrix = result.sliding(matrixBTest.length, matrixBTest.length).toArray

    if (!similar(resultMatrix, resultTest)){
      println("Expected: ")
      println(resultTest.deep.mkString("\n"))
      println("Result: ")
      println(resultMatrix.deep.mkString("\n"))
    }

    assert(similar(resultMatrix, resultTest))
  }
}

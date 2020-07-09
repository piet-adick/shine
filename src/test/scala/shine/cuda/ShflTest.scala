package shine.cuda

import shine.DPIA.NatIdentifier
import shine.DPIA.Phrases._
import shine.DPIA.Types._
import shine.cuda.primitives.functional.ShflDown

class ShflTest extends shine.test_util.Tests {

  test("ShflDown test") {
    val arr = Identifier("arr", ExpType(ArrayType(32, f32), read))
    val delta = NatIdentifier("delta")
    val x = Identifier("x", ExpType(ArrayType(32, f32), read))
    val shflDownTest =
      DepLambda[NatKind](delta)(
        Lambda(arr,
          ShflDown(f32, delta, x)
        )
      )

    val code = KernelGenerator().makeCode(shflDownTest).code
    println(code)

  }

}

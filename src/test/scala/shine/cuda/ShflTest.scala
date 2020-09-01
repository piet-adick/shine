package shine.cuda

import rise.core.primitives.Split
import shine.DPIA.{Nat, NatIdentifier}
import shine.DPIA.Phrases._
import shine.DPIA.Types.DataType.idx
import shine.DPIA.Types._
import shine.cuda.primitives.functional._

class ShflTest extends shine.test_util.Tests {

  test("ShflDown test") {
    val in = Identifier("arr", ExpType(ArrayType(32, f32), read))
    val srcLanes = Identifier("arr", ExpType(ArrayType(32, idx(32:Nat)), read))
    val delta = NatIdentifier("delta")

    //split(32) |> mapWarp(fun(x => x |> split(1) |> toPrivateFun(mapLane(id)) |> let(fun(x => zip(x, Shfl(x)))) |> toPrivateFun(mapLane(addOp)))

    val shflDownTest =
      DepLambda[NatKind](delta)(
        Lambda(in,
          Lambda(srcLanes,
            ShflWarp(f32, srcLanes , in)
          )
        )
      )

    val code = KernelGenerator().makeCode(shflDownTest).code
    println(code)

  }

}

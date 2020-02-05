package shine.examples

import shine.C.ProgramGenerator
import shine.DPIA.Phrases._
import shine.DPIA.Types._
import shine.DPIA.Types.AddressSpace._
import shine.DPIA.FunctionalPrimitives._
import shine.DPIA.Semantics.OperationalSemantics.FloatData
import shine.DPIA._
import shine.OpenCL.FunctionalPrimitives.{MapGlobal, OpenCLReduceSeq, To}
import shine.OpenCL.KernelGenerator

class matmul extends test_util.Tests {

  test("matmul") {
    val x = Identifier(freshName("x"), ExpType(PairType(float, float), read))
    val y = Identifier(freshName("y"), ExpType(float, read))
    val z = Identifier(freshName("z"), ExpType(float, read))

    val n = NatIdentifier(freshName("n"))
    val rowA = Identifier(freshName("vecA"), ExpType(ArrayType(n, float), read))
    val columnB = Identifier(freshName("vecB"), ExpType(ArrayType(n, float), read))
    val matA = Identifier(freshName("matA"), ExpType(ArrayType(n, ArrayType(n, float)), read))
    val matB = Identifier(freshName("matB"), ExpType(ArrayType(n, ArrayType(n, float)), read))

    val mul = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, Fst(float, float, x), Snd(float, float, x)))
    val add = Lambda[ExpType, FunType[ExpType, ExpType]](y, Lambda[ExpType, ExpType](z, BinOp(Operators.Binary.ADD, y, z)))

    val matvec = DepLambda[NatKind](n)(Lambda[ExpType, FunType[ExpType, ExpType]](rowA, Lambda[ExpType, ExpType](matB,
      MapSeq(n, ArrayType(n, float), float,
        Lambda[ExpType, ExpType] (columnB,
          ReduceSeq(n, float, float, add, Literal(FloatData(0.0f)),
            MapSeq(n, PairType(float, float), float, mul,
              Zip(n, float, float, rowA, columnB))))
        , Transpose(n, n, float, matB))
    )))

    val matmul = DepLambda[NatKind](n)(Lambda[ExpType, FunType[ExpType, ExpType]](matA, Lambda[ExpType, ExpType](matB,
      MapSeq(n, ArrayType(n, float), ArrayType(n, float),
        Lambda[ExpType, ExpType] (rowA,
          MapSeq(n, ArrayType(n, float), float,
            Lambda[ExpType, ExpType] (columnB,
              ReduceSeq(n, float, float, add, Literal(FloatData(0.0f)),
                MapSeq(n, PairType(float, float), float, mul,
                  Zip(n, float, float, rowA, columnB)))),
            Transpose(n, n, float, matB))), matA
      ))))


    println(ProgramGenerator.makeCode(matvec, "Matvec").code)
    println(ProgramGenerator.makeCode(matmul, "Matmul").code)
  }

  test("matmul OpenCL") {
    val x = Identifier(freshName("x"), ExpType(PairType(float, float), read))
    val y = Identifier(freshName("y"), ExpType(float, read))
    val z = Identifier(freshName("z"), ExpType(float, read))

    val n = NatIdentifier(freshName("n"))
    val rowA = Identifier(freshName("vecA"), ExpType(ArrayType(n, float), read))
    val columnB = Identifier(freshName("vecB"), ExpType(ArrayType(n, float), read))
    val matA = Identifier(freshName("matA"), ExpType(ArrayType(n, ArrayType(n, float)), read))
    val matB = Identifier(freshName("matB"), ExpType(ArrayType(n, ArrayType(n, float)), read))

    val mul = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, Fst(float, float, x), Snd(float, float, x)))
    val add = Lambda[ExpType, FunType[ExpType, ExpType]](y, Lambda[ExpType, ExpType](z, BinOp(Operators.Binary.ADD, y, z)))

    val matvec = DepLambda[NatKind](n)(Lambda[ExpType, FunType[ExpType, ExpType]](rowA, Lambda[ExpType, ExpType](matB,
      MapGlobal(0)(n, ArrayType(n, float), float,
        Lambda[ExpType, ExpType] (columnB,
          OpenCLReduceSeq(n, Global, float, float, add, Literal(FloatData(0.0f)),
            To(Global, ArrayType(n, float),
              MapGlobal(0)(n, PairType(float, float), float, mul,
                Zip(n, float, float, rowA, columnB))), false))
        , matB)
    )))

    val matmul = DepLambda[NatKind](n)(Lambda[ExpType, FunType[ExpType, ExpType]](matA, Lambda[ExpType, ExpType](matB,
      MapGlobal(0)(n, ArrayType(n, float), ArrayType(n, float),
        Lambda[ExpType, ExpType] (rowA,
          MapGlobal(0)(n, ArrayType(n, float), float,
            Lambda[ExpType, ExpType] (columnB,
              OpenCLReduceSeq(n, Global, float, float, add, Literal(FloatData(0.0f)),
                To(Global, ArrayType(n, float),
                  MapGlobal(0)(n, PairType(float, float), float, mul,
                    Zip(n, float, float, rowA, columnB))), false))
            , matB))
        , matA
      ))))

    println(KernelGenerator.makeCode(matvec, "Matvec").code)
    println(KernelGenerator.makeCode(matmul, "Matmul").code)
  }
}

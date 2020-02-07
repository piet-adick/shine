package shine.examples

import shine.C.ProgramGenerator
import shine.DPIA.Phrases._
import shine.DPIA.Types._
//import shine.DPIA.Types.AddressSpace._
import shine.DPIA.FunctionalPrimitives._
import shine.DPIA.Semantics.OperationalSemantics.FloatData
import shine.DPIA._
//import shine.OpenCL.FunctionalPrimitives.{MapGlobal, OpenCLReduceSeq, To}
//import shine.OpenCL.KernelGenerator

class matrixmultiplication_nxm_mxr extends test_util.Tests {

  test("matrixMult") {


    // def dotproduct = fun((v,w) => reduce(+, 0) <<: map(*) <<: zip(v,w)) // rise
    // def matrixmultiplication = fun((v,w)) => map(map(dotproduct(v,transpose(w)))  => transpose(w)

    //def add(a: float, b: float):float = a + b

    val x = Identifier(freshName("x"), ExpType(PairType(float, float), read))
    val y = Identifier(freshName("y"), ExpType(float, read))
    val z = Identifier(freshName("z"), ExpType(float, read))

    val n = NatIdentifier(freshName("n"))
    val m = NatIdentifier(freshName("m"))
    val r = NatIdentifier(freshName("r"))
    val columnB = Identifier(freshName("columnB"), ExpType(ArrayType(m, float), read))
    val rowA = Identifier(freshName("rowA"), ExpType(ArrayType(m, float), read))
    val matrixA = Identifier(freshName("MatrixA"), ExpType(ArrayType(n, ArrayType(m,float)),read))
    val matrixB = Identifier(freshName("MatrixB"), ExpType(ArrayType(m, ArrayType(r,float)),read))

    val mul = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, Fst(float, float, x), Snd(float, float, x)))
    val add = Lambda[ExpType, FunType[ExpType, ExpType]](y, Lambda[ExpType, ExpType](z, BinOp(Operators.Binary.ADD, y, z)))

    val dotproduct = Lambda[ExpType, ExpType](rowA,
      ReduceSeq(m, float, float, add, Literal(FloatData(0.0f)),
        MapSeq(m, PairType(float, float), float, mul,
          Zip(m, float, float, rowA, columnB))))

    val matrixMult = DepLambda[NatKind](r)(DepLambda[NatKind](m)(DepLambda[NatKind](n)(
      Lambda[ExpType, FunType[ExpType, ExpType]](
      matrixA, Lambda[ExpType, ExpType](matrixB,
        MapSeq(n, ArrayType(m,float), ArrayType(r,float),
          Lambda[ExpType, ExpType](columnB,
            MapSeq(r, ArrayType(m,float), float,
              dotproduct,
              Transpose(m, r, float,matrixB))),
          matrixA))
    ))))
    println(ProgramGenerator.makeCode(matrixMult, "matrixMult").code)

  }


}
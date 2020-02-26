package shine.examples

import shine.C.ProgramGenerator
import shine.DPIA.FunctionalPrimitives.{Fst, MapSeq, ReduceSeq, Snd, Transpose, Zip}
import shine.DPIA.Phrases.{BinOp, DepLambda, Identifier, Lambda, Literal, Operators}
import shine.DPIA.Semantics.OperationalSemantics.FloatData
import shine.DPIA.Types.{ArrayType, ExpType, FunType, NatKind, PairType, f32, read}
import shine.DPIA.{NatIdentifier, freshName}
import shine.test_util

//GEMM = general matrix multiply
//With A,B,C matrices
//This calculate result = A*B + C
class gemm extends test_util.Tests {

  val x = Identifier(freshName("x"), ExpType(PairType(f32, f32), read))
  val y = Identifier(freshName("y"), ExpType(f32, read))
  val z = Identifier(freshName("z"), ExpType(f32, read))

  val n = NatIdentifier(freshName("n"))
  val m = NatIdentifier(freshName("m"))
  val k = NatIdentifier(freshName("k"))

  //pair with row of MatrixA and row of matrix C
  val rowAC = Identifier(freshName("vecAC"), ExpType(PairType(ArrayType(m, f32),ArrayType(k, f32)), read))
  //piar with column of MatrixB and single float of matrix C
  val columnBFC = Identifier(freshName("columnBFC"), ExpType(PairType(ArrayType(m, f32), f32), read))
  val matA = Identifier(freshName("matA"), ExpType(ArrayType(n, ArrayType(m, f32)), read))
  val matB = Identifier(freshName("matB"), ExpType(ArrayType(m, ArrayType(k, f32)), read))
  val matC = Identifier(freshName("matC"), ExpType(ArrayType(n, ArrayType(k, f32)), read))

  val mul = Lambda[ExpType, ExpType](x, BinOp(Operators.Binary.MUL, Fst(f32, f32, x), Snd(f32, f32, x)))
  val add = Lambda[ExpType, FunType[ExpType, ExpType]](y, Lambda[ExpType, ExpType](z, BinOp(Operators.Binary.ADD, y, z)))

  test("GEMM C") {
    val gemm = DepLambda[NatKind](n)(DepLambda[NatKind](m)(DepLambda[NatKind](k)(
      Lambda[ExpType, FunType[ExpType, FunType[ExpType, ExpType]]](matC,
        Lambda[ExpType, FunType[ExpType, ExpType]](matA,
          Lambda[ExpType, ExpType](matB,
            MapSeq(n, PairType(ArrayType(m, f32), ArrayType(k, f32)), ArrayType(k, f32),
              Lambda[ExpType, ExpType] (rowAC,
                MapSeq(k, PairType(ArrayType(m, f32), f32), f32,
                  Lambda[ExpType, ExpType] (columnBFC,
                    BinOp(Operators.Binary.ADD,
                      ReduceSeq(m, f32, f32, add, Literal(FloatData(0.0f)),
                        MapSeq(m, PairType(f32, f32), f32, mul,
                          Zip(m, f32, f32,
                            Fst(ArrayType(m,f32), ArrayType(k, f32), rowAC),
                            Fst(ArrayType(m,f32), f32, columnBFC)))),
                      Snd(ArrayType(m,f32), f32, columnBFC))),
                  Zip(k, ArrayType(m, f32), f32,
                    Transpose(m, k, f32, matB), Snd(ArrayType(m,f32), ArrayType(k, f32), rowAC)))),
              Zip(n, ArrayType(m, f32), ArrayType(k, f32), matA, matC))))))))

    println(ProgramGenerator.makeCode(gemm, "gemm").code)
  }
}

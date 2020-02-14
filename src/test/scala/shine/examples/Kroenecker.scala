package shine.examples

import shine.C.ProgramGenerator
import shine.DPIA.Phrases.{BinOp, Identifier, IfThenElse, Lambda, Literal, Operators}
import shine.DPIA.Semantics.OperationalSemantics.FloatData
import shine.DPIA.Types.{ExpType, FunType, float, read}
import shine.DPIA.freshName
//import shine.DPIA.{NatIdentifier, freshName}

class Kroenecker extends test_util.Tests{

  test("maximum"){

    //def kroenecker = fun(a,b) => a==b ? 1 : 0
    val a = Identifier(freshName("a"), ExpType(float, read))
    val b = Identifier(freshName("b"), ExpType(float, read))
    val max = Lambda[ExpType, FunType[ExpType, ExpType]](a, Lambda[ExpType, ExpType](b, IfThenElse(BinOp(Operators.Binary.GT, a,b), a, b)))

    println(ProgramGenerator.makeCode(max, "max").code)
  }

  test("minimum"){

    //def kroenecker = fun(a,b) => a==b ? 1 : 0
    val a = Identifier(freshName("a"), ExpType(float, read))
    val b = Identifier(freshName("b"), ExpType(float, read))
    val min = Lambda[ExpType, FunType[ExpType, ExpType]](a, Lambda[ExpType, ExpType](b, IfThenElse(BinOp(Operators.Binary.LT, a,b), a, b)))

    println(ProgramGenerator.makeCode(min, "min").code)
  }

  test("kroenecker-umstaendlich"){

    //def kroenecker = fun(a,b) => a==b ? 1 : 0
    val a = Identifier(freshName("a"), ExpType(float, read))
    val b = Identifier(freshName("b"), ExpType(float, read))
    /*
    bisher ist der Test noch umständlich, weil die Funktionen jeweils Phrases erwaten
    und nicht einfach ein Float-Wert 1.0f als Eingabe, also als Ergebnis akzeptieren
     */
    val kroenecker = Lambda[ExpType, FunType[ExpType, ExpType]](a, Lambda[ExpType, ExpType](b, IfThenElse(BinOp(Operators.Binary.EQ, a,b), BinOp(Operators.Binary.DIV, a,a), BinOp(Operators.Binary.SUB, a,a))))

    println(ProgramGenerator.makeCode(kroenecker, "kroenecker").code)
  }

  test("kroenecker"){

    //def kroenecker = fun(a,b) => a==b ? 1 : 0
    val a = Identifier(freshName("a"), ExpType(float, read))
    val b = Identifier(freshName("b"), ExpType(float, read))
    /*
    bisher ist der Test noch umständlich, weil die Funktionen jeweils Phrases erwaten
    und nicht einfach ein Float-Wert 1.0f als Eingabe, also als Ergebnis akzeptieren
     */
    val kroenecker = Lambda[ExpType, FunType[ExpType, ExpType]](a, Lambda[ExpType, ExpType](b, IfThenElse(BinOp(Operators.Binary.EQ, a,b), Literal(FloatData(1.0f)), Literal(FloatData(0.0f)))))

    println(ProgramGenerator.makeCode(kroenecker, "kroenecker").code)
  }
}

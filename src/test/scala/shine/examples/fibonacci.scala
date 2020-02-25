package shine.examples

import shine.C.ProgramGenerator
import shine.test_util
//import shine.DPIA.ImperativePrimitives.For
import shine.DPIA.Phrases.{BinOp, Identifier, IfThenElse, Lambda, Literal, Natural, Operators}
import shine.DPIA.Semantics.OperationalSemantics.{FloatData}
import shine.DPIA.Types.{ExpType, NatType, read}
import shine.DPIA.{freshName}

class fibonacci extends test_util.Tests{
  //calculate the fibonacci-number
  test("fibonacci"){
    //f0= 0,f1= 1
    val fib0 = Literal(FloatData(0.0f));
    val fib1 = Literal(FloatData(1.0f));

    val n = Identifier(freshName("n"), ExpType(NatType, read))
    //val m = NatIdentifier(freshName("m"))
    //fn=fn−1+fn−2f ̈urn= 2,3,4,...
    //the following comment out Code can't work, because
    //NatData and IndexData are not accepted in Literal
    //Phrase line 120 assert(!d.isInstanceOf[NatData])
    /*val fib = Lambda[ExpType, ExpType](n,
      IfThenElse(BinOp(Operators.Binary.EQ, Literal(NatData(0)),Literal(NatData(0))), fib0,
        fib1)
    )*/

    //fn=1, falls n!=0, sonst fn=0 (f0=0)
    val fib_vereinfacht = Lambda[ExpType, ExpType](n,
      IfThenElse(BinOp(Operators.Binary.EQ, n,Natural(0)), fib0,
        fib1)
    )

    //fn=1, falls n==1,fn=0, falls n==0, sonst fn=8
    val fib_vereinfacht2 = Lambda[ExpType, ExpType](n,
      IfThenElse(BinOp(Operators.Binary.EQ, n,Natural(0)), fib0,
        IfThenElse(BinOp(Operators.Binary.EQ, n,Natural(1)), fib1,
          Literal(FloatData(8))))
    )



/*
recursion is probably not wanted, if I use this fib_rec method,
then is the error: recursive value fib_rec needs type fib_rec))
 */
    /*val fib_rec =
      IfThenElse(BinOp(Operators.Binary.EQ, n,Natural(0)), fib0,
        IfThenElse(BinOp(Operators.Binary.EQ, n,Natural(1)), fib1,
          fib_rec))
*/
    val fib_rec =
      IfThenElse(BinOp(Operators.Binary.EQ, n,Natural(0)), fib0,
        IfThenElse(BinOp(Operators.Binary.EQ, n,Natural(1)), fib1,
          Literal(FloatData(8))))

    val fib_rufeFktauf = Lambda[ExpType, ExpType](n,
      IfThenElse(BinOp(Operators.Binary.EQ, n,Natural(0)), fib0,
        IfThenElse(BinOp(Operators.Binary.EQ, n,Natural(1)), fib1,
          fib_rec))
    )

    //no idea yet how For works
    /*val minus1 = BinOp(Operators.Binary.SUB, n , Natural(1))
    val bodyFkt = Lambda[ExpType, CommType](n, BinOp(Operators.Binary.EQ, n,Natural(0)))
    val countTo100 = Lambda[ExpType, ExpType](n,
      For(m,  , Literal(BoolData(true) ))
    )*/

    //I need to know how for works, because I could implement every function
    //like fibonacci, which works with recursion, with for-loops

    println(ProgramGenerator.makeCode(fib0, "fib0").code)
    println(ProgramGenerator.makeCode(fib_vereinfacht, "fib_vereinfacht").code)
    println(ProgramGenerator.makeCode(fib_vereinfacht2, "fib_vereinfacht2").code)
    println(ProgramGenerator.makeCode(fib_rufeFktauf, "fib_rufeFktauf").code)
  }
}

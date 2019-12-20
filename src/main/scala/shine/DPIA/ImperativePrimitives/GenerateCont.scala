package shine.DPIA.ImperativePrimitives

import shine.DPIA.Compilation.TranslationContext
import shine.DPIA._
import shine.DPIA.Types._
import shine.DPIA.Phrases._
import shine.DPIA.Semantics.OperationalSemantics._

import scala.xml.Elem

//noinspection ScalaUnnecessaryParentheses
// note: would not be necessary if generate was defined as indices + map
final case class GenerateCont(n: Nat,
                              dt: DataType,
                              f: Phrase[ExpType ->: ((ExpType ->: CommType) ->: CommType)])
  extends ExpPrimitive
{
  override val t: ExpType =
    (n: Nat) ->: (dt: DataType) ->:
      (f :: exp"[idx($n), $read]" ->: t"exp[$dt, $read] -> comm" ->: comm) ->:
      exp"[$n.$dt, $read]"

  override def visitAndRebuild(v: VisitAndRebuild.Visitor): Phrase[ExpType] = {
    GenerateCont(v.nat(n), v.data(dt), VisitAndRebuild(f, v))
  }

  override def eval(s: Store): Data = ???

  override def acceptorTranslation(A: Phrase[AccType])
                                  (implicit context: TranslationContext): Phrase[CommType] =
    throw new Exception("This should not happen")

  override def continuationTranslation(C: Phrase[->:[ExpType, CommType]])
                                      (implicit context: TranslationContext): Phrase[CommType] =
    throw new Exception("This should not happen")

  override def prettyPrint: String = s"(generateCont $n $f)"

  override def xmlPrinter: Elem =
    <generateCont n={ToString(n)} dt={ToString(dt)}>
      <f>
        {Phrases.xmlPrinter(f)}
      </f>
    </generateCont>
}
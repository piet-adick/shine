package shine.DPIA.ImperativePrimitives

import shine.DPIA._
import shine.DPIA.DSL._
import shine.DPIA.Types._
import shine.DPIA.Phrases._
import shine.DPIA.Semantics.OperationalSemantics.{AccIdentifier, Store}

final case class CycleAcc(n: Nat,
                          m: Nat,
                          dt: DataType,
                          input: Phrase[AccType])
  extends AccPrimitive
{
  override val t: AccType =
    (n: Nat) ->: (m: Nat) ->: (dt: DataType) ->:
      (input :: acc"[$m.$dt]") ->: acc"[$n.$dt]"

  override def eval(s: Store): AccIdentifier = ???

  override def visitAndRebuild(v: VisitAndRebuild.Visitor): Phrase[AccType] =
    CycleAcc(v.nat(n), v.nat(m), v.data(dt), VisitAndRebuild(input, v))

  override def xmlPrinter: xml.Elem =
    <cycleAcc n={ToString(n)} m={ToString(m)} dt={ToString(dt)}>
      {Phrases.xmlPrinter(input)}
    </cycleAcc>

  override def prettyPrint: String = s"(cycleAcc $input)"
}
package shine.cuda.primitives.functional

import shine.DPIA.Compilation.TranslationContext
import shine.DPIA.DSL._
import shine.DPIA.Phrases.VisitAndRebuild.Visitor
import shine.DPIA.Phrases._
import shine.DPIA.Semantics.OperationalSemantics.{Data, Store}
import shine.DPIA.Types._
import shine.DPIA.Types.DataType._
import shine.DPIA._
import shine.{cuda => c}

import scala.xml.Elem

final case class ShflUpWarp(
                               dt: ScalarType,
                               delta: Nat,
                               in: Phrase[ExpType]
                             )
  extends ExpPrimitive
{
  val warpSize: Nat = c.warpSize

  in :: expT((32:Nat)`.`dt, read)
  override val t: ExpType = expT((32:Nat)`.`dt, read)

  override def visitAndRebuild(f: Visitor): Phrase[ExpType] =
    ShflUpWarp(f.data(dt), f.nat(delta), VisitAndRebuild(in, f))

  override def prettyPrint: String =
    s"ShflUpWarp($delta, ${PrettyPhrasePrinter(in)}"

  def acceptorTranslation(A: Phrase[AccType])
                         (implicit context: TranslationContext): Phrase[CommType] = ???

  def continuationTranslation(C: Phrase[ExpType ->: CommType])
                             (implicit context: TranslationContext): Phrase[CommType] =
  {
    import shine.DPIA.Compilation.TranslationToImperative._
    con(in)(λ(expT(warpSize`.`dt, read))(inImp =>
      C(ShflUpWarp(dt, delta, inImp))
    ))
  }

  override def eval(s: Store): Data = ???

  override def xmlPrinter: Elem = <shflDownWarp />

}
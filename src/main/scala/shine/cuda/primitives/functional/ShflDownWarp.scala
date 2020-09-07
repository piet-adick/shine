package shine.cuda.primitives.functional

import shine.DPIA.Compilation.TranslationContext
import shine.DPIA.DSL._
import shine.DPIA.Phrases.VisitAndRebuild.Visitor
import shine.DPIA.Phrases._
import shine.DPIA.Semantics.OperationalSemantics.{Data, IndexData, Store}
import shine.DPIA.Types._
import shine.DPIA.Types.DataType._
import shine.DPIA._
import shine.{cuda => c}
import shine.cuda.primitives.imperative.ShflDownWarpSync

import scala.xml.Elem

final case class ShflDownWarp(
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
    ShflDownWarp(f.data(dt), delta, VisitAndRebuild(in, f))

  override def prettyPrint: String =
    s"ShflDownWarp($delta, ${PrettyPhrasePrinter(in)}"

  def acceptorTranslation(A: Phrase[AccType])
                         (implicit context: TranslationContext): Phrase[CommType] = ???

  def continuationTranslation(C: Phrase[ExpType ->: CommType])
                             (implicit context: TranslationContext): Phrase[CommType] =
    {
      import shine.DPIA.Compilation.TranslationToImperative._
      con(in)(λ(expT(warpSize`.`dt, read))(inImp =>
        C(ShflDownWarpSync(0xFFFFFFFF, dt, delta, inImp`@`Literal(IndexData(0, 1))))
      ))
    }


  override def eval(s: Store): Data = ???

  override def xmlPrinter: Elem = ???

}
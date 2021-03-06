package shine.cuda.primitives.functional

import shine.DPIA.Compilation.TranslationContext
import shine.DPIA.DSL._
import shine.DPIA.Phrases.VisitAndRebuild.Visitor
import shine.DPIA.Phrases._
import shine.DPIA.Semantics.OperationalSemantics.{Data, Store}
import shine.DPIA.Types._
import shine.DPIA.Types.DataType._
import shine.DPIA._

import scala.xml.Elem

final case class ShflUp(
  dt: ScalarType,
  delta: Nat,
  in: Phrase[ExpType]
)
  extends ExpPrimitive
{
  in :: expT((32:Nat)`.`dt, read)
  override val t: ExpType = expT((32:Nat)`.`dt, read)

  override def visitAndRebuild(f: Visitor): Phrase[ExpType] =
    ShflUp(f.data(dt), delta, VisitAndRebuild(in, f))

  override def eval(s: Store): Data = ???

  override def prettyPrint: String = ???

  override def xmlPrinter: Elem = ???

  def acceptorTranslation(A: Phrase[AccType])
                         (implicit context: TranslationContext): Phrase[CommType] =
  {
    import shine.DPIA.Compilation.TranslationToImperative._
    con(in)(λ(expT((32:Nat)`.`dt, read))(inImp =>
      A :=|((32:Nat)`.`dt)| ShflUp(dt, delta, inImp)
    ))
  }

  def continuationTranslation(C: Phrase[ExpType ->: CommType])
                             (implicit context: TranslationContext): Phrase[CommType] =
  {
    import shine.DPIA.Compilation.TranslationToImperative._
    con(in)(λ(expT((32:Nat)`.`dt, read))(inImp =>
      C(ShflUp(dt, delta, inImp))
    ))
  }

}
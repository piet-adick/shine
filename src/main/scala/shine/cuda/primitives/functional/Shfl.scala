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

final case class Shfl(
  dt: ScalarType,
  srcLanes: Phrase[ExpType],
  in: Phrase[ExpType]
)
  extends ExpPrimitive
{
  srcLanes :: expT((32:Nat)`.`idx((32:Nat)), read)
  in :: expT((32:Nat)`.`dt, read)
  override val t: ExpType = expT((32:Nat)`.`dt, read)

  override def visitAndRebuild(f: Visitor): Phrase[ExpType] =
    Shfl(f.data(dt), VisitAndRebuild(srcLanes, f), VisitAndRebuild(in, f))

  override def eval(s: Store): Data = ???

  override def prettyPrint: String = ???

  override def xmlPrinter: Elem = ???

  def acceptorTranslation(A: Phrase[AccType])
                         (implicit context: TranslationContext): Phrase[CommType] =
    {
      import shine.DPIA.Compilation.TranslationToImperative._
      con(srcLanes)(λ(expT((32:Nat)`.`idx((32:Nat)), read))(srcLanesImp =>
        con(in)(λ(expT((32:Nat)`.`dt, read))(inImp =>
          A :=|((32:Nat)`.`dt)| Shfl(dt, srcLanesImp, inImp)
        ))
      ))
    }

  def continuationTranslation(C: Phrase[ExpType ->: CommType])
                             (implicit context: TranslationContext): Phrase[CommType] =
    {
      import shine.DPIA.Compilation.TranslationToImperative._
      con(srcLanes)(λ(expT((32:Nat)`.`idx((32:Nat)), read))(srcLanesImp =>
        con(in)(λ(expT((32:Nat)`.`dt, read))(inImp =>
          C(Shfl(dt, srcLanesImp, inImp))
        ))
      ))
    }

}
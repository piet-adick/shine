package shine.cuda.primitives.functional

import shine.DPIA.Compilation.TranslationContext
import shine.DPIA.DSL._
import shine.DPIA.Phrases.VisitAndRebuild.Visitor
import shine.DPIA.Phrases._
import shine.DPIA.Semantics.OperationalSemantics.{Data, IndexData, Store}
import shine.DPIA.Types._
import shine.DPIA.Types.DataType._
import shine.DPIA._
import shine.cuda.primitives.imperative.ShflXorWarpSync

import scala.xml.Elem

final case class ShflXorWarp(
  dt: ScalarType,
  laneMask: Phrase[ExpType],
  in: Phrase[ExpType]
)
  extends ExpPrimitive
{
  laneMask :: expT(idx(32:Nat), read)
  in :: expT((32:Nat)`.`dt, read)
  override val t: ExpType = expT((32:Nat)`.`dt, read)

  override def visitAndRebuild(f: Visitor): Phrase[ExpType] =
    ShflWarp(f.data(dt), VisitAndRebuild(laneMask, f), VisitAndRebuild(in, f))

  override def prettyPrint: String = ???

  def acceptorTranslation(A: Phrase[AccType])
                         (implicit context: TranslationContext): Phrase[CommType] = ???

  def continuationTranslation(C: Phrase[ExpType ->: CommType])
                             (implicit context: TranslationContext): Phrase[CommType] =
    {
      import shine.DPIA.Compilation.TranslationToImperative._
      con(laneMask)(λ(expT(idx(32:Nat), read))(maskImp =>
        con(in)(λ(expT((32:Nat)`.`dt, read))(inImp =>
          C(ShflXorWarpSync(0xFFFFFFFF, dt, maskImp, inImp`@`Literal(IndexData(0, 1))))
        ))
      ))
    }


  override def eval(s: Store): Data = ???

  override def xmlPrinter: Elem = ???

}
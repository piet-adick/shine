package shine.cuda.primitives.functional

import shine.DPIA.Compilation.TranslationContext
import shine.DPIA.DSL._
import shine.DPIA.Phrases.VisitAndRebuild.Visitor
import shine.DPIA.Phrases._
import shine.DPIA.Semantics.OperationalSemantics.{Data, IndexData, Store}
import shine.DPIA.Types._
import shine.DPIA.Types.DataType._
import shine.DPIA._
import shine.cuda.laneId
import shine.cuda.primitives.imperative.ShflWarpSync

import scala.xml.Elem

final case class ShflWarp(
  dt: ScalarType, //TODO change ScalarType to ShflType since some vector types are also allowed
  srcLanes: Phrase[ExpType],
  in: Phrase[ExpType]
)
  extends ExpPrimitive
{
  srcLanes :: expT((32:Nat)`.`idx(32:Nat), read)
  in :: expT((32:Nat)`.`dt, read)
  override val t: ExpType = expT((32:Nat)`.`dt, read)

  override def visitAndRebuild(f: Visitor): Phrase[ExpType] =
    ShflWarp(f.data(dt), VisitAndRebuild(srcLanes, f), VisitAndRebuild(in, f))

  override def prettyPrint: String =
    s"ShflWarp(${PrettyPhrasePrinter(srcLanes)}, ${PrettyPhrasePrinter(in)}"

  def acceptorTranslation(A: Phrase[AccType])
                         (implicit context: TranslationContext): Phrase[CommType] = ???

  def continuationTranslation(C: Phrase[ExpType ->: CommType])
                             (implicit context: TranslationContext): Phrase[CommType] =
    {
      import shine.DPIA.Compilation.TranslationToImperative._
      con(srcLanes)(λ(expT((32:Nat)`.`idx(32:Nat), read))(srcLanesImp =>
        con(in)(λ(expT((32:Nat)`.`dt, read))(inImp =>
          C(ShflWarpSync(0xFFFFFFFF, dt, srcLanesImp`@`laneId('x'), inImp`@`Literal(IndexData(0, 1))))
        ))
      ))
    }


  override def eval(s: Store): Data = ???

  override def xmlPrinter: Elem = ???

}
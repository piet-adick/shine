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

final case class ShflWarp(
  dt: ScalarType, //TODO change ScalarType to ShflType since some vector types are also allowed
  srcLanes: Phrase[ExpType],
  in: Phrase[ExpType]
)
  extends ExpPrimitive
{
  val warpSize: Nat = c.warpSize

  srcLanes :: expT(warpSize`.`idx(warpSize), read)
  in :: expT(warpSize`.`dt, read)
  override val t: ExpType = expT(warpSize`.`dt, read)

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
      con(srcLanes)(λ(expT(warpSize`.`idx(warpSize), read))(srcLanesImp =>
        con(in)(λ(expT(warpSize`.`dt, read))(inImp =>
          C(ShflWarp(dt, srcLanesImp, inImp))
        ))
      ))
    }


  override def eval(s: Store): Data = ???

  override def xmlPrinter: Elem = <shflWarp />

}
package shine.cuda.primitives.imperative

import shine.DPIA.{->:, Nat, expT}
import shine.DPIA.Compilation.TranslationContext
import shine.DPIA.DSL._
import shine.DPIA.Phrases._
import shine.DPIA.Semantics.OperationalSemantics
import shine.DPIA.Semantics.OperationalSemantics.Store
import shine.DPIA.Types.DataType.idx
import shine.DPIA.Types._

import scala.xml.Elem

final case class ShflWarpSync(
  mask: Nat,
  dt: ScalarType,
  srcLane: Phrase[ExpType],
  value: Phrase[ExpType]
) extends ExpPrimitive {

  srcLane :: expT(idx(32:Nat), read)
  value :: expT(dt, read)

  override val t: ExpType = expT(dt, read)

  override def visitAndRebuild(f: VisitAndRebuild.Visitor): Phrase[ExpType] =
    ShflWarpSync(f.nat(mask), f.data(dt), VisitAndRebuild(srcLane, f), VisitAndRebuild(value, f))

  //FIXME
  override def prettyPrint: String = s"shflWarpSync()"

  override def acceptorTranslation(
    A: Phrase[AccType]
  )(implicit context: TranslationContext): Phrase[CommType] = ???

  override def continuationTranslation(
    C: Phrase[ExpType ->: CommType]
  )(implicit context: TranslationContext): Phrase[CommType] = {
    import shine.DPIA.Compilation.TranslationToImperative._

    con(srcLane)(fun(expT(idx(32:Nat), read))(srcLaneImp =>
      con(value)(fun(expT(dt, read))(valueImp =>
        C(ShflWarpSync(mask, dt, srcLaneImp, valueImp))
      ))
    ))
  }



  override def eval(s: Store): OperationalSemantics.Data = ???

  override def xmlPrinter: Elem = ???
}

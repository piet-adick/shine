package shine.cuda.primitives.imperative

import shine.DPIA.{->:, Nat, expT}
import shine.DPIA.Compilation.TranslationContext
import shine.DPIA.DSL._
import shine.DPIA.Phrases._
import shine.DPIA.Semantics.OperationalSemantics
import shine.DPIA.Semantics.OperationalSemantics.Store
import shine.DPIA.Types._

import scala.xml.Elem

final case class ShflDownWarpSync(
  mask: Nat,
  dt: ScalarType,
  delta: Nat,
  value: Phrase[ExpType]
) extends ExpPrimitive {

  value :: expT(dt, read)

  override val t: ExpType = expT(dt, read)

  override def visitAndRebuild(f: VisitAndRebuild.Visitor): Phrase[ExpType] =
    ShflDownWarpSync(f.nat(mask), f.data(dt), f.nat(delta), VisitAndRebuild(value, f))

  //FIXME
  override def prettyPrint: String = s"shflDownWarpSync()"

  override def acceptorTranslation(
    A: Phrase[AccType]
  )(implicit context: TranslationContext): Phrase[CommType] = ???

  override def continuationTranslation(
    C: Phrase[ExpType ->: CommType]
  )(implicit context: TranslationContext): Phrase[CommType] = {
    import shine.DPIA.Compilation.TranslationToImperative._

    con(value)(fun(expT(dt, read))(valueImp =>
      C(ShflDownWarpSync(mask, dt, delta, valueImp))
    ))
  }


  override def eval(s: Store): OperationalSemantics.Data = ???

  override def xmlPrinter: Elem = ???

}

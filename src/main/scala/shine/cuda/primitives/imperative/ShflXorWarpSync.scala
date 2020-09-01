package shine.cuda.primitives.imperative

import shine.DPIA.{->:, Nat, expT}
import shine.DPIA.Compilation.TranslationContext
import shine.DPIA.Phrases.{ExpPrimitive, Phrase, VisitAndRebuild}
import shine.DPIA.Semantics.OperationalSemantics
import shine.DPIA.Semantics.OperationalSemantics.Store
import shine.DPIA.Types.DataType.idx
import shine.DPIA.Types._

import scala.xml.Elem

final case class ShflXorWarpSync(
  mask : Nat,
  dt: ScalarType,
  laneMask: Phrase[ExpType],
  value: Phrase[ExpType]
) extends ExpPrimitive {
  laneMask :: expT(idx(32:Nat), read)
  value :: expT(dt, read)

  override val t: ExpType = expT(dt, read)

  override def visitAndRebuild(f: VisitAndRebuild.Visitor): Phrase[ExpType] = ???

  override def acceptorTranslation(A: Phrase[AccType])(implicit context: TranslationContext): Phrase[CommType] = ???

  override def continuationTranslation(C: Phrase[ExpType ->: CommType])(implicit context: TranslationContext): Phrase[CommType] = ???

  override def prettyPrint: String = ???


  override def eval(s: Store): OperationalSemantics.Data = ???

  override def xmlPrinter: Elem = ???

}

package shine.cuda.primitives.functional

import shine.DPIA.Compilation.TranslationContext
import shine.DPIA.Phrases.VisitAndRebuild.Visitor
import shine.DPIA.Phrases._
import shine.DPIA.Semantics.OperationalSemantics.{Data, Store}
import shine.DPIA.Types._
import shine.DPIA.Types.DataType._
import shine.DPIA._

import scala.xml.Elem

// CUDA: T __shfl_sync (unsigned int mask, T var, int srcLane, int width=warpSize)
// Imperative: shflSync32: (mask : nat) → (t : number) → idx[32] → t → t
// Functional: shfl: (t : number) → 32.idx[32] → 32.t → 32.t

final case class Shfl(
                     dt: ScalarType,
                     msk: Phrase[ExpType],
                     in: Phrase[ExpType]
                     )
  extends ExpPrimitive
{

  msk :: expT((32:Nat)`.`idx((32:Nat)), read)
  in :: expT((32:Nat)`.`dt, read)
  override val t: ExpType = expT((32:Nat)`.`dt, read)

  override def visitAndRebuild(f: Visitor): Phrase[ExpType] = ???

  override def eval(s: Store): Data = ???

  override def prettyPrint: String = ???

  override def xmlPrinter: Elem = ???

  def acceptorTranslation(A: Phrase[AccType])
                         (implicit context: TranslationContext): Phrase[CommType] = ???

  def continuationTranslation(C: Phrase[ExpType ->: CommType])
                             (implicit context: TranslationContext): Phrase[CommType] = ???

}

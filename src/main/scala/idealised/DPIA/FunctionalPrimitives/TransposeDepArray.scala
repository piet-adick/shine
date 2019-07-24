package idealised.DPIA.FunctionalPrimitives

import idealised.DPIA.Compilation.{TranslationContext, TranslationToImperative}
import idealised.DPIA.DSL._
import idealised.DPIA.Phrases._
import idealised.DPIA.Semantics.OperationalSemantics._
import idealised.DPIA.Types._
import idealised.DPIA._

import scala.xml.Elem

final case class TransposeDepArray(n:Nat,
                                   m:Nat,
                                   f:NatToData,
                                   array:Phrase[ExpType])
  extends ExpPrimitive {

  override val t: ExpType = {
    (n: Nat) ->: (m: Nat) ->:
      (array :: exp"[$n.${DepArrayType(m, k => f(k))}, $read]") ->:
      exp"[${DepArrayType(m, k => ArrayType(n, f(k)))}, $read]"
  }

  override def visitAndRebuild(v: VisitAndRebuild.Visitor): Phrase[ExpType] = {
    TransposeDepArray(v.nat(n), v.nat(m), v.natToData(f), VisitAndRebuild(array, v))
  }

  override def acceptorTranslation(A: Phrase[AccType])(implicit context: TranslationContext): Phrase[CommType] = {
    ???
  }

  override def mapAcceptorTranslation(f: Phrase[ExpType ->: ExpType], A: Phrase[AccType])(implicit context: TranslationContext): Phrase[CommType] = ???


  override def continuationTranslation(C: Phrase[ExpType ->: CommType])(implicit context: TranslationContext): Phrase[CommType] = {
    import TranslationToImperative._
    con(array)(λ(exp"[$n.${DepArrayType(m, k => f(k))}, $read]")(x => C(TransposeDepArray(n, m, f, x))))
  }

  override def xmlPrinter: Elem = {
    <transposeArrayDep n={ToString(n)} m={ToString(m)} f={ToString(f)}>
      {Phrases.xmlPrinter(array)}
    </transposeArrayDep>
  }

  override def prettyPrint: String = s"(transposeArrayDep $n $m $f ${PrettyPhrasePrinter(array)}"

  override def eval(s: Store): Data = ???
}

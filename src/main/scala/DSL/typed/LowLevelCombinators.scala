package DSL.typed

import Core.TypeInference._
import Core._
import LowLevelCombinators._
import OpenCL.Core.OpenCLAddressSpace$
import apart.arithmetic.ArithExpr

object `new` {
  def apply(dt: DataType,
            addressSpace: AddressSpace,
            f: Phrase[(ExpType x AccType) -> CommandType]) =
    New(dt, addressSpace, f)

  def apply(dt: DataType,
            addressSpace: AddressSpace,
            f: Phrase[ExpType x AccType] => Phrase[CommandType]) =
    New(dt, addressSpace, λ(exp"[$dt]" x acc"[$dt]")( v => f(v) ))
}

object `if` {
  def apply[T <: PhraseType](cond: Phrase[ExpType],
                             thenP: Phrase[T],
                             elseP: Phrase[T]) =
    IfThenElsePhrase(cond, thenP, elseP)
}

object `for` {
  def apply(n: ArithExpr,
            f: (Phrase[ExpType] => Phrase[CommandType])) =
    For(n, λ(exp"[$int]")( i => f(i) ))
}

object `parFor` {
  def apply(n: ArithExpr,
            dt: DataType,
            out: Phrase[AccType],
            f: (Phrase[ExpType] => Phrase[AccType] => Phrase[CommandType])) =
    ParFor(n, dt, out, λ(exp"[$int]")( i => λ(acc"[$dt]")( o => f(i)(o) )))
}

object dblBufFor {
  def apply(n: ArithExpr,
            dt: DataType,
            addressSpace: AddressSpace,
            buffer1: Phrase[VarType],
            buffer2: Phrase[VarType],
            k: ArithExpr,
            body: Phrase[`(nat)->`[AccType -> (ExpType -> CommandType)]],
            C: Phrase[ExpType -> CommandType]) =
    DoubleBufferFor(n, dt, addressSpace, buffer1, buffer2, k, body, C)
}

object fst {
  def apply(record: Phrase[ExpType]) = {
    record.t match {
      case ExpType(RecordType(dt1, dt2)) => Fst(dt1, dt2, record)
      case x => error(x.toString, "ExpType(RecordType)")
    }
  }
}

object snd {
  def apply(record: Phrase[ExpType]) = {
    record.t match {
      case ExpType(RecordType(dt1, dt2)) => Snd(dt1, dt2, record)
      case x => error(x.toString, "ExpType(RecordType)")
    }
  }
}

object fstAcc {
  def apply(fstT: DataType, sndT: DataType, record: Phrase[AccType]) =
    FstAcc(fstT, sndT, record)
}

object sndAcc {
  def apply(fstT: DataType, sndT: DataType, record: Phrase[AccType]) =
    SndAcc(fstT, sndT, record)
}

object skip extends Skip
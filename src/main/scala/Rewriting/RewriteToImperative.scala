package Rewriting

import Core._
import Core.PhraseType._
import DSL._
import ExpPatterns._
import AccPatterns._
import CommandPatterns._

object RewriteToImperative {

  def apply(p: Phrase[ExpType -> ExpType]): Phrase[CommandType] = {
    val in = identifier.exp("input")
    val out = identifier.acc("output")
    acc(p(in), out)
  }

  private def acc(E: Phrase[ExpType], A: Phrase[AccType]): Phrase[CommandType] = {
    E match {
      case x: IdentPhrase[ExpType] if x.t.dataType.isBasicType =>
        A `:=` x

      case x: IdentPhrase[ExpType] if x.t.dataType.isInstanceOf[ArrayType] =>
        MapI(A,
          λ(A.t) { o => λ(x.t) { x => acc(x, o) } },
          x
        )

      case x: IdentPhrase[ExpType] if x.t.dataType.isInstanceOf[RecordType] =>
        acc(fst(x), fstAcc(A)) `;` acc(snd(x), sndAcc(A))

      case c : LiteralPhrase => A `:=` c

      case BinOpPhrase(op, e1, e2) =>
        exp(e1, λ(e1.t) { x =>
          exp(e2, λ(e2.t) { y =>
            A `:=` BinOpPhrase(op, x, y)
          })
        })

      case pattern : ExpPattern => pattern match {

        case Map(f, e) =>
          exp(e, λ(e.t) { x =>
            MapI(A,
              λ(A.t) { o => λ(e.t) { x => acc(f(x), o) } },
              x
            )
          })

        case Reduce(f, i, e) =>
          exp(e, λ(e.t) { x =>
            exp(i, λ(i.t) { y =>
              ReduceIAcc(A,
                λ(A.t) { o => λ(e.t) { x => λ(i.t) { y => acc(f(x)(y), o) } } },
                y,
                x
              )
            })
          })

        case Zip(e1, e2) =>
          exp(e1, λ(e1.t) { x =>
            exp(e2, λ(e2.t) { y =>
              MapI(A,
                λ(A.t) { o => λ(ExpType(RecordType(e1.t.dataType, e2.t.dataType))) { x => acc(x, o) } },
                Zip(x, y)
              )
            })
          })

        case Join(e) => acc(e, JoinAcc(A))

        case Split(n, e) => acc(e, SplitAcc(n, A))

      }

      // on the fly beta-reduction
      case ApplyPhrase(fun, arg) => acc(Lift.liftFunction(fun)(arg), A)
    }
  }

  private def exp(E: Phrase[ExpType], C: Phrase[ExpType -> CommandType]): Phrase[CommandType] = {
    E match {
      case x: IdentPhrase[ExpType] => C(x)

      case c : LiteralPhrase => C(c)

      case BinOpPhrase(op, e1, e2) =>
        exp(e1, λ(e1.t) { x =>
          exp(e2, λ(e2.t) { y =>
            C(BinOpPhrase(op, x, y))
          })
        })

      case pattern : ExpPattern => pattern match {

        case Map(f, e) =>
          // specify array type + size info
          `new`( tmp =>
            acc(Map(f, e), π2(tmp)) `;`
            C(π1(tmp))
          )

        case Reduce(f, i, e) =>
          exp(e, λ(e.t) { x =>
            exp(i, λ(i.t) { y =>
              ReduceIExp(C,
                λ(AccType(i.t.dataType)) { o => λ(e.t) { x => λ(i.t) { y => acc(f(x)(y), o) } } },
                y,
                x
              )
            })
          })

        case Zip(e1, e2) =>
          exp(e1, λ(e1.t) { x =>
            exp(e2, λ(e2.t) { y =>
              C(Zip(x, y))
            })
          })

        case Join(e) =>
          exp(e, λ(e.t) { x =>
            C(Join(x))
          })

        case Split(n, e) =>
          exp(e, λ(e.t) { x =>
            C(Split(n, x))
          })

      }

      // on the fly beta-reduction
      case ApplyPhrase(fun, arg) => exp(Lift.liftFunction(fun)(arg), C)

      case IfThenElsePhrase(cond, thenP, elseP) => throw new Exception("This should never happen")
      case Proj1Phrase(pair) => throw new Exception("This should never happen")
      case Proj2Phrase(pair) => throw new Exception("This should never happen")
    }
  }

}

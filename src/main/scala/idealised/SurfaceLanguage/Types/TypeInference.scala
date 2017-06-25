package idealised.SurfaceLanguage.Types

import idealised.SurfaceLanguage._

import scala.language.existentials

class TypeInferenceException(expr: String, msg: String)
  extends TypeException(s"Failed to infer type for `$expr'. $msg.")

object TypeInference {
  def error(expr: String, found: String, expected: String): Nothing = {
    throw new TypeInferenceException(expr, s"Found $found but expected $expected")
  }

  def error(expr: String, msg: String): Nothing = {
    throw new TypeInferenceException(expr, msg)
  }

  type SubstitutionMap = scala.collection.Map[Expr[_], Expr[_]]

  def apply[T <: Type](expr: Expr[T], subs: SubstitutionMap): Expr[T] = {
    inferType(expr, subs)
  }

  private def inferType[T <: Type](expr: Expr[T], subs: SubstitutionMap): Expr[T] = {
    (expr match {
      case i@IdentifierExpr(name, t) =>
        val identifier =
          if (subs.contains(i)) {
            subs(i)
          } else {
            i
          }
        identifier.`type` match {
          case Some(_) => identifier
          case None => error(identifier.toString, s"Found Identifier $name without type")
        }

      case LambdaExpr(param, body) =>
        inferType(param, subs) match {
          case newParam: IdentifierExpr =>
            LambdaExpr(newParam, inferType(body, subs))
          case _ => throw new Exception("")
        }

      case ApplyExpr(fun, arg) => ApplyExpr(inferType(fun, subs), inferType(arg, subs))

      case NatDependentLambdaExpr(x, e) =>
        NatDependentLambdaExpr(x, inferType(e, subs))

      case NatDependentApplyExpr(f, x) =>
        NatDependentApplyExpr(inferType(f, subs), x)

      case TypeDependentLambdaExpr(x, e) =>
        TypeDependentLambdaExpr(x, inferType(e, subs))

      case TypeDependentApplyExpr(f, x) =>
        TypeDependentApplyExpr(inferType(f, subs), x)

      case IfThenElseExpr(cond, thenE, elseE) =>
        IfThenElseExpr(inferType(cond, subs), inferType(thenE, subs), inferType(elseE, subs))

      case UnaryOpExpr(op, e) => UnaryOpExpr(op, inferType(e, subs))

      case BinOpExpr(op, lhs, rhs) => BinOpExpr(op, inferType(lhs, subs), inferType(rhs, subs))

      case LiteralExpr(d, dt) => LiteralExpr(d, dt)

      case p: PrimitiveExpr => p.inferType(subs)
    }).asInstanceOf[Expr[T]]
  }

  def setParamAndInferType[T <: Type](f: Expr[DataType -> T],
                                      t: DataType,
                                      subs: SubstitutionMap): Expr[DataType -> T] = {
    f match {
      case LambdaExpr(x, e) =>
        val newX = IdentifierExpr(newName(), Some(t))
        val newE = apply(e, subs.updated(x, newX))
        LambdaExpr(newX, newE)
      case _ => throw new Exception("This should not happen")
    }
  }

  def setParamsAndInferTypes[T <: Type](f: Expr[DataType -> (DataType -> T)],
                                        t1: DataType,
                                        t2: DataType,
                                        subs: SubstitutionMap): Expr[DataType -> (DataType -> T)] = {
    f match {
      case LambdaExpr(x, e1) =>
        val newX = IdentifierExpr(newName(), Some(t1))
        e1 match {
          case LambdaExpr(y, e2) =>
            val newY = IdentifierExpr(newName(), Some(t2))
            val newE2 = apply(e2, subs.updated(x, newX).updated(y, newY))
            LambdaExpr(newX, LambdaExpr(newY, newE2))
          case _ => throw new Exception("This should not happen")
        }
      case _ => throw new Exception("This should not happen")
    }
  }
}
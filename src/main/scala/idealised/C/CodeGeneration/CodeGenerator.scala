package idealised.C.CodeGeneration

import idealised.C.AST.{Block, Node}
import idealised.DPIA.DSL._
import idealised.DPIA.FunctionalPrimitives._
import idealised.DPIA.ImperativePrimitives._
import idealised.DPIA.Phrases._
import idealised.DPIA.Semantics.OperationalSemantics
import idealised.DPIA.Semantics.OperationalSemantics._
import idealised.DPIA.Types._
import idealised.DPIA._
import idealised.SurfaceLanguage.Operators
import idealised._
import lift.arithmetic.BoolExpr.ArithPredicate
import lift.arithmetic.{NamedVar, _}

import scala.collection.immutable.VectorBuilder
import scala.collection.{immutable, mutable}
import scala.language.implicitConversions

object CodeGenerator {

  final case class Environment(identEnv: immutable.Map[Identifier[_ <: BasePhraseTypes], C.AST.DeclRef],
                               commEnv: immutable.Map[Identifier[CommandType], C.AST.Stmt],
                               contEnv: immutable.Map[Identifier[ExpType -> CommandType], Phrase[ExpType] => Environment => C.AST.Stmt],
                               inlLetNatEnv: immutable.Map[LetNatIdentifier, C.AST.Expr]
                              ) {
    def updatedIdentEnv(kv: (Identifier[_ <: BasePhraseTypes], C.AST.DeclRef)): Environment = {
      this.copy(identEnv = identEnv + kv)
    }

    def updatedCommEnv(kv: (Identifier[CommandType], C.AST.Stmt)): Environment = {
      this.copy(commEnv = commEnv + kv)
    }

    def updatedContEnv(kv: (Identifier[ExpType -> CommandType], Phrase[ExpType] => Environment => C.AST.Stmt)): Environment = {
      this.copy(contEnv = contEnv + kv)
    }

    def updatedInlNatEnv(kv:(LetNatIdentifier, C.AST.Expr)):Environment = {
      this.copy(inlLetNatEnv = this.inlLetNatEnv + kv)
    }
  }

  object Environment {
    def empty = Environment(
      immutable.Map(),
      immutable.Map(),
      immutable.Map(),
      immutable.Map()
    )
  }

  sealed trait PathExpr
  sealed trait TupleAccess extends PathExpr
  final case object FstMember extends TupleAccess
  final case object SndMember extends TupleAccess
  final case class CIntExpr(num: Nat) extends PathExpr
  implicit def cIntExprToNat(cexpr: CIntExpr): Nat = cexpr.num

  type Path = immutable.List[PathExpr]

  type Declarations = mutable.ListBuffer[C.AST.Decl]
  type Ranges = immutable.Map[String, lift.arithmetic.Range]

  def apply(): CodeGenerator =
    new CodeGenerator(mutable.ListBuffer[C.AST.Decl](), immutable.Map[String, lift.arithmetic.Range]())
}

class CodeGenerator(val decls: CodeGenerator.Declarations,
                    val ranges: CodeGenerator.Ranges)
  extends DPIA.Compilation.CodeGenerator[CodeGenerator.Environment, CodeGenerator.Path, C.AST.Stmt, C.AST.Expr, C.AST.Decl, C.AST.DeclRef, C.AST.Type] {

  import CodeGenerator._

  type Environment = CodeGenerator.Environment
  type Path = CodeGenerator.Path

  type Stmt = C.AST.Stmt
  type Decl = C.AST.Decl
  type Expr = C.AST.Expr
  type Ident = C.AST.DeclRef
  type Type = C.AST.Type

  override def name: String = "C"

  def addDeclaration(decl: Decl): Unit = {
    if (decls.exists(_.name == decl.name)) {
      println(s"warning: declaration with name ${decl.name} already defined")
    } else {
      decls += decl
    }
  }


  def updatedRanges(key: String, value: lift.arithmetic.Range): CodeGenerator =
    new CodeGenerator(decls, ranges.updated(key, value))

  override def generate(phrase:Phrase[CommandType],
               topLevelDefinitions:scala.Seq[(LetNatIdentifier, Phrase[ExpType])],
               env:CodeGenerator.Environment): (scala.Seq[Decl], Stmt) = {
    val stmt = this.generateWithFunctions(phrase, topLevelDefinitions, env)
    (decls, stmt)
  }

  def generateWithFunctions(phrase:Phrase[CommandType],
                            topLevelDefinitions:scala.Seq[(LetNatIdentifier, Phrase[ExpType])],
                            env:CodeGenerator.Environment):Stmt = {
    topLevelDefinitions.headOption match {
      case Some((ident, defn)) =>
        generateLetNat(ident, defn, env, (gen, env) => gen.generateWithFunctions(phrase, topLevelDefinitions.tail, env))
      case None => cmd(phrase,env)
    }
  }

  override def cmd(phrase: Phrase[CommandType], env: Environment): Stmt = {
    visitAndGenerateNat(phrase match {
      case Phrases.IfThenElse(cond, thenP, elseP) =>
        exp(cond, env, Nil, cond =>
          C.AST.IfThenElse(cond, cmd(thenP, env), Some(cmd(elseP, env))))

      case i: Identifier[CommandType] => env.commEnv(i)

      case Apply(i: Identifier[_], e) => // TODO: think about this
        env.contEnv(
          i.asInstanceOf[Identifier[ExpType -> CommandType]]
        )(
          e.asInstanceOf[Phrase[ExpType]]
        )(env)

      case Skip() => C.AST.Comment("skip")

      case Seq(p1, p2) => C.AST.Stmts(cmd(p1, env), cmd(p2, env))

      case Assign(_, a, e) =>
        exp(e, env, Nil, e =>
          acc(a, env, Nil, a =>
            C.AST.ExprStmt(C.AST.Assignment(a, e))))

      case New(dt, Lambda(v, p)) => CCodeGen.codeGenNew(dt, v, p, env)

      case NewDoubleBuffer(_, _, dt, n, in, out, Lambda(ps, p)) =>
        CCodeGen.codeGenNewDoubleBuffer(ArrayType(n, dt), in, out, ps, p, env)

      case NewRegRot(n, dt, Lambda(registers, Lambda(rotate, body))) =>
        CCodeGen.codeGenNewRegRot(n, dt, registers, rotate, body, env)

      case For(n, Lambda(i, p), unroll) => CCodeGen.codeGenFor(n, i, p, unroll, env)

      case ForNat(n, DepLambda(i: NatIdentifier, p), unroll) => CCodeGen.codeGenForNat(n, i, p, unroll, env)

      case Proj1(pair) => cmd(Lifting.liftPair(pair)._1, env)
      case Proj2(pair) => cmd(Lifting.liftPair(pair)._2, env)

      case LetNat(binder, defn, body) => generateLetNat(binder, defn, env, (gen, env) => gen.cmd(body, env))


      case Apply(_, _) | DepApply(_, _) |
           _: CommandPrimitive =>
        error(s"Don't know how to generate code for $phrase")
    }, env)
  }

  override def acc(phrase: Phrase[AccType],
                   env: Environment,
                   path: Path,
                   cont: Expr => Stmt): Stmt = {
    phrase match {
      case i@Identifier(_, AccType(dt)) => cont(CCodeGen.generateAccess(dt,
        env.identEnv.applyOrElse(i, (_: Phrase[_]) => {
          throw new Exception(s"Expected to find `$i' in the environment: `${env.identEnv}'")
        }), path, env))

      case SplitAcc(n, _, _, a) => path match {
        case (i : CIntExpr) :: ps  => acc(a, env, CIntExpr(i / n) :: CIntExpr(i % n) :: ps, cont)
        case _ => error(s"Expected a C-Integer-Expression on the path.")
      }
      case JoinAcc(_, m, _, a) => path match {
        case (i : CIntExpr) :: (j : CIntExpr) :: ps => acc(a, env, CIntExpr(i * m + j) :: ps, cont)
        case _ => error(s"Expected two C-Integer-Expressions on the path.")
      }
      case depJ@DepJoinAcc(_, _, _, a) => path match {
        case (i : CIntExpr) :: (j : CIntExpr) :: ps =>
          acc(a, env, CIntExpr(BigSum(0, i - 1, x => depJ.lenF(x)) + j) :: ps, cont)
        case _ => error(s"Expected two C-Integer-Expressions on the path.")
      }

      case RecordAcc1(_, _, a) => acc(a, env, FstMember :: path, cont)
      case RecordAcc2(_, _, a) => acc(a, env, SndMember :: path, cont)

      case ZipAcc1(_, _, _, a) => path match {
        case (i : CIntExpr) :: ps => acc(a, env, i :: FstMember :: ps, cont)
        case _ => error(s"Expected a C-Integer-Expression on the path.")
      }
      case ZipAcc2(_, _, _, a) => path match {
        case (i : CIntExpr) :: ps => acc(a, env, i :: SndMember :: ps, cont)
        case _ => error(s"Expected a C-Integer-Expression on the path.")
      }
      case UnzipAcc(_, _, _, _) => ???

      case TakeAcc(_, _, _, a) => acc(a, env, path, cont)
      case DropAcc(n, _, _, a) => path match {
        case (i : CIntExpr) :: ps => acc(a, env, CIntExpr(i + n) ::ps, cont)
        case _ => error(s"Expected a C-Integer-Expression on the path.")
      }

      case CycleAcc(_, m, _, a) => path match {
        case (i : CIntExpr) :: ps => acc(a, env, CIntExpr(i % m) :: ps, cont)
        case _ => error(s"Expected a C-Integer-Expression on the path.")
      }

      case ReorderAcc(n, _, idxF, a) => path match {
        case (i : CIntExpr) :: ps =>
          acc(a, env, CIntExpr(OperationalSemantics.evalIndexExp(idxF(AsIndex(n, Natural(i))))) :: ps, cont)
        case _ => error(s"Expected a C-Integer-Expression on the path.")
      }

      case MapAcc(n, dt, _, f, a) => path match {
        case (i : CIntExpr) :: ps => acc( f( IdxAcc(n, dt, AsIndex(n, Natural(i)), a) ), env, ps, cont)
        case _ => error(s"Expected a C-Integer-Expression on the path.")
      }

      case IdxAcc(_, _, i, a) => CCodeGen.codeGenIdxAcc(i, a, env, path, cont)

      case DepIdxAcc(_, _, i, a) => acc(a, env, CIntExpr(i) :: path, cont)

      case Proj1(pair) => acc(Lifting.liftPair(pair)._1, env, path, cont)
      case Proj2(pair) => acc(Lifting.liftPair(pair)._2, env, path, cont)

      case Apply(_, _) | DepApply(_, _) |
           Phrases.IfThenElse(_, _, _) | _: AccPrimitive =>
        error(s"Don't know how to generate code for $phrase")
    }
  }

  override def exp(phrase: Phrase[ExpType],
                   env: Environment,
                   path: Path,
                   cont: Expr => Stmt) : Stmt =
  {
    phrase match {
      case i@Identifier(_, ExpType(dt)) => cont(CCodeGen.generateAccess(dt,
        env.identEnv.applyOrElse(i, (_: Phrase[_]) => {
          throw new Exception(s"Expected to find `$i' in the environment: `${env.identEnv}'")
        }), path, env))

      case Phrases.Literal(n) => cont(path match {
        case Nil =>
            n.dataType match {
              case _: IndexType => CCodeGen.codeGenLiteral(n)
              case _: ScalarType => CCodeGen.codeGenLiteral(n)
              case _ => error ("Expected an IndexType or ScalarType.")
          }
        case (i : CIntExpr) :: Nil =>
          n match {
            case SingletonArrayData(_, a) => CCodeGen.codeGenLiteral(a)
            case _ =>
              n.dataType match {
                case _: ArrayType => C.AST.ArraySubscript(CCodeGen.codeGenLiteral(n), C.AST.ArithmeticExpr(i))
                case _ => error("Expected an ArrayType.")
              }
          }
        // case (_ :: _ :: Nil, _: ArrayType) => C.AST.Literal("0.0f") // TODO: (used in gemm like this) !!!!!!!
        case _ => error(s"Unexpected: $n $path")
      })

      case Phrases.Natural(n) => cont(path match {
        case Nil => C.AST.ArithmeticExpr(n)
        case _ => error(s"Expected the path to be empty.")
      })

      case uop@UnaryOp(op, e) => uop.t.dataType match {
        case _: ScalarType => path match {
          case Nil => exp(e, env, Nil, e =>
            cont(CCodeGen.codeGenUnaryOp(op, e)))
          case _ => error(s"Expected path to be empty")
        }
        case _ => error(s"Expected scalar types")
      }

      case bop@BinOp(op, e1, e2) => bop.t.dataType match {
        case _: ScalarType => path match {
          case Nil =>
            exp(e1, env, Nil, e1 =>
              exp(e2, env, Nil, e2 =>
                cont(CCodeGen.codeGenBinaryOp(op, e1, e2))))
          case _ => error(s"Expected path to be empty")
        }
        case _ => error(s"Expected scalar types")
      }

      case Cast(_, dt, e) => path match {
        case Nil =>
          exp(e, env, Nil, e =>
            cont(C.AST.Cast(typ(dt), e)))
        case _ => error(s"Expected path to be empty")
      }

      case IndexAsNat(_, e) => exp(e, env, path, cont)

      case AsIndex(_, e) => exp(e, env, path, cont)

      case Split(n, _, _, e) => path match {
        case (i : CIntExpr) :: (j : CIntExpr) :: ps => exp(e, env, CIntExpr(n * i + j) :: ps, cont)
        case _ => error(s"Expected two C-Integer-Expressions on the path.")
      }
      case Join(_, m, _, e) => path match {
        case (i : CIntExpr) :: ps => exp(e, env, CIntExpr(i / m) :: CIntExpr(i % m) :: ps, cont)
        case _ => error(s"Expected two C-Integer-Expressions on the path.")
      }

      case part@Partition(_, _, _, _, e) => path match {
        case (i: CIntExpr) :: (j: CIntExpr) :: ps =>
          exp(e, env, CIntExpr(BigSum(0, i - 1, x => part.lenF(x)) + j) :: ps, cont)
        case _ => error(s"Expected path to contain at least two elements")
      }

      case Zip(_, _, _, e1, e2) => path match {
        case (i: CIntExpr) :: (xj : TupleAccess) :: ps => xj match {
          case FstMember => exp(e1, env, i :: ps, cont)
          case SndMember => exp(e2, env, i :: ps, cont)
        }
        case _ => error("Expected a C-Integer-Expression followed by a tuple access on the path.")
      }
      case Unzip(_, _, _, _) => ???

      case Record(_, _, e1, e2) => path match {
        case (xj : TupleAccess) :: ps => xj match {
          case FstMember => exp(e1, env, ps, cont)
          case SndMember => exp(e2, env, ps, cont)
        }
        case _ => error("Expected a tuple access on the path.")
      }
      case Fst(_, _, e) => exp(e, env, FstMember :: path, cont)
      case Snd(_, _, e) => exp(e, env, SndMember :: path, cont)

      case Take(_, _, _, e) => exp(e, env, path, cont)

      case Drop(n, _, _, e) => path match {
        case (i : CIntExpr) :: ps => exp(e, env, CIntExpr(i + n) :: ps, cont)
        case _ => error(s"Expected a C-Integer-Expression on the path.")
      }

      case Cycle(_, m, _, e) => path match {
        case (i : CIntExpr) :: ps => exp(e, env, CIntExpr(i % m) :: ps, cont)
        case _ => error(s"Expected a C-Integer-Expression on the path.")
      }

      case Reorder(n, _, idxF, _, a) => path match {
        case (i : CIntExpr) :: ps =>
          exp(a, env, CIntExpr(OperationalSemantics.evalIndexExp(idxF(AsIndex(n, Natural(i))))) :: ps, cont)
        case _ => error(s"Expected a C-Integer-Expression on the path.")
      }

      case PadClamp(n, l, r, _, e) => path match {
        case (i: CIntExpr) :: ps =>
          exp(e, env, CIntExpr(0) :: ps, left =>
            exp(e, env, CIntExpr(n-1) :: ps, right =>
            genPad(n, l, r, left, right, i, ps, e, env, cont)))
      }

      case Pad(n, l, r, _, pad, array) => path match {
        case (i: CIntExpr) :: ps =>
          exp(pad, env, ps, padExpr =>
            genPad(n, l, r, padExpr, padExpr, i, ps, array, env, cont))

        case _ => error(s"Expected path to be not empty")
      }

      case Slide(_, _, s2, _, e) => path match {
        case (i : CIntExpr) :: (j : CIntExpr) :: ps => exp(e, env, CIntExpr(i * s2 + j) :: ps, cont)
        case _ => error(s"Expected two C-Integer-Expressions on the path.")
      }

      case TransposeDepArray(_, _, _, _, e) => path match {
        case (i : CIntExpr)::(j : CIntExpr) :: ps => exp(e, env, CIntExpr(j) :: CIntExpr(i) :: ps , cont)
        case _ => error(s"Expected two C-Integer-Expressions on the path.")
      }

      case MapRead(n, dt1, dt2, f, e) => path match {
        case (i : CIntExpr) :: ps =>
          val continue_cmd =
            Identifier[ExpType -> CommandType](s"continue_$freshName",
              FunctionType(ExpType(dt2), comm))

          cmd(f(
            Idx(n, dt1, AsIndex(n, Natural(i)), e)
          )(
            continue_cmd
          ), env updatedContEnv (continue_cmd -> (e => env => exp(e, env, ps, cont))))
        case _ => error(s"Expected path to be not empty")
      }

      case GenerateCont(n, dt, f) => path match {
        case (i : CIntExpr) :: ps =>
          val continue_cmd =
            Identifier[ExpType -> CommandType](s"continue_$freshName",
              FunctionType(ExpType(dt), comm))

          cmd(f(AsIndex(n, Natural(i)))(continue_cmd),
            env updatedContEnv (continue_cmd -> (e => env => exp(e, env, ps, cont))))
        case _ => error(s"Expected path to be not empty")
      }

      case Idx(_, _, i, e) => CCodeGen.codeGenIdx(i, e, env, path, cont)

      case DepIdx(_, _, i, e) => exp(e, env, CIntExpr(i) :: path, cont)

      case ForeignFunction(f, inTs, outT, args) =>
        CCodeGen.codeGenForeignFunction(f, inTs, outT, args, env, path, cont)

      case Proj1(pair) => exp(Lifting.liftPair(pair)._1, env, path, cont)
      case Proj2(pair) => exp(Lifting.liftPair(pair)._2, env, path, cont)

      case Apply(_, _) | DepApply(_, _) |
           Phrases.IfThenElse(_, _, _) | _: ExpPrimitive =>
        error(s"Don't know how to generate code for $phrase")
    }
  }

  override def typ(dt: DataType): Type = {

    def typeToName(t:DataType):String = {
      t match {
        case IndexType(_) => freshName("idx")
        case _ => t.toString
      }
    }

    dt match {
      case b: idealised.DPIA.Types.BasicType => b match {
        case idealised.DPIA.Types.bool => C.AST.Type.int
        case idealised.DPIA.Types.int | idealised.DPIA.Types.NatType => C.AST.Type.int
        case idealised.DPIA.Types.float => C.AST.Type.float
        case idealised.DPIA.Types.double => C.AST.Type.double
        case _: idealised.DPIA.Types.IndexType => C.AST.Type.int
      }
      case a: idealised.DPIA.Types.ArrayType => C.AST.ArrayType(typ(a.elemType), Some(a.size))
      case a: idealised.DPIA.Types.DepArrayType => C.AST.ArrayType(typ(a.elemFType.body), Some(a.size)) // TODO: be more precise with the size?
      case r: idealised.DPIA.Types.RecordType =>
        C.AST.StructType(typeToName(r.fst) + "_" + typeToName(r.snd), immutable.Seq(
          (typ(r.fst), "_fst"),
          (typ(r.snd), "_snd")))
      case _: idealised.DPIA.Types.DataTypeIdentifier => throw new Exception("This should not happen")
    }
  }


  private def generateLetNat[T <: PhraseType](binder:LetNatIdentifier,
                             defn:Phrase[T],
                             env:Environment,
                             cont:(CodeGenerator ,Environment) => Stmt):Stmt = {
    defn.t match {
      case ExpType(_) =>
        exp(defn.asInstanceOf[Phrase[ExpType]], env, List(), e => cont(this, env updatedInlNatEnv(binder, e)))
      case _ =>
        val newCodeGen = defineNatFunction(binder.name, defn, env)
        cont(newCodeGen, env)
    }
  }

  private def defineNatFunction[T <: PhraseType](name: String, phrase: Phrase[T], env: Environment): CodeGenerator = {

    def getPhraseAndParams[_ <: PhraseType](p: Phrase[_],
                                            ps: immutable.Seq[Identifier[ExpType]] = immutable.Seq(),
                                            ranges:immutable.Seq[(String, Range)] = immutable.Seq()
                                           ): (Phrase[ExpType], immutable.Seq[Identifier[ExpType]], immutable.Seq[(String, Range)]) = {
      p match {
        case l: Lambda[ExpType, _]@unchecked => getPhraseAndParams(l.body, l.param +: ps, ranges)
        case ndl: NatDependentLambda[_] => getPhraseAndParams(ndl.body, Identifier(ndl.x.name, ExpType(int)) +: ps, (ndl.x.name, ndl.x.range) +: ranges)
        case ep: Phrase[ExpType]@unchecked => (ep, ps.reverse, ranges.reverse)
      }
    }

    val (body, params, ranges) = getPhraseAndParams(phrase)

    val newEnv = params.foldLeft(env)((e, ident) => e.updatedIdentEnv(ident, C.AST.DeclRef(ident.name)))

    val newCodeGen = ranges.foldLeft(this)((gen, rangeInfo) => gen.updatedRanges(rangeInfo._1, rangeInfo._2))

    val cBody = visitAndGenerateNat(newCodeGen.exp(body, newEnv, List(), C.AST.Return(_)), newEnv)
    val decl = C.AST.FunDecl(
      name,
      typ(body.t.dataType),
      params.map(ident => C.AST.ParamDecl(ident.name, typ(ident.t.dataType))),
      C.AST.Block(immutable.Seq(cBody))
    )
    newCodeGen.addDeclaration(decl)
    newCodeGen
  }

  /**
    * This function takes a phrase with a nat dependent free variable `for`, and it generates a block
    * where `for` is bound to the arithmetic expression at.
    * @param `for` The free nat variable to substitute
    * @param phrase The phrase to generate
    * @param at The arithmetic expression we are generating phrase at
    * @param env Up-to-date environment
    * @return
    */
  protected def generateNatDependentBody(`for`: NatIdentifier,
                                       phrase: Phrase[CommandType],
                                       at: ArithExpr,
                                       env: Environment): Block = {
    PhraseType.substitute(at, `for`, in = phrase) |> (p => {
      val newIdentEnv = env.identEnv.map {
        case (Identifier(name, AccType(dt)), declRef) =>
          (Identifier(name, AccType(DataType.substitute(at, `for`, in = dt))), declRef)
        case (Identifier(name, ExpType(dt)), declRef) =>
          (Identifier(name, ExpType(DataType.substitute(at, `for`, in = dt))), declRef)
        case x => x
      }
      C.AST.Block(immutable.Seq(this.cmd(p, env.copy(identEnv = newIdentEnv))))
    })
  }

  protected object CCodeGen {
    def codeGenNew(dt: DataType,
                   v: Identifier[VarType],
                   p: Phrase[CommandType],
                   env: Environment): Stmt = {
      val ve = Identifier(s"${v.name}_e", v.t.t1)
      val va = Identifier(s"${v.name}_a", v.t.t2)
      val vC = C.AST.DeclRef(v.name)

      C.AST.Block(immutable.Seq(
        C.AST.DeclStmt(C.AST.VarDecl(vC.name, typ(dt))),
        cmd(Phrase.substitute(Pair(ve, va), `for` = v, `in` = p),
          env updatedIdentEnv (ve -> vC)
            updatedIdentEnv (va -> vC))))
    }

    def codeGenNewDoubleBuffer(dt: ArrayType,
                               in: Phrase[ExpType],
                               out: Phrase[AccType],
                               ps: Identifier[VarType x CommandType x CommandType],
                               p: Phrase[CommandType],
                               env: Environment): Stmt = {
      import C.AST._
      import BinaryOperator._
      import UnaryOperator._

      val ve = Identifier(s"${ps.name}_e", ps.t.t1.t1.t1)
      val va = Identifier(s"${ps.name}_a", ps.t.t1.t1.t2)
      val done = Identifier(s"${ps.name}_swap", ps.t.t1.t2)
      val swap = Identifier(s"${ps.name}_done", ps.t.t2)

      val tmp1 = DeclRef(freshName("tmp1_"))
      val tmp2 = DeclRef(freshName("tmp2_"))
      val in_ptr = DeclRef(freshName("in_ptr_"))
      val out_ptr = DeclRef(freshName("out_ptr_"))
      val flag = DeclRef(freshName("flag_"))

      Block(immutable.Seq(
        // create variables: `tmp1', `tmp2`, `in_ptr', and `out_ptr'
        DeclStmt(VarDecl(tmp1.name, typ(dt))),
        DeclStmt(VarDecl(tmp2.name, typ(dt))),
        exp(in, env, CIntExpr(0) :: Nil, e => makePointerDecl(in_ptr.name, dt.elemType, UnaryExpr(&, e))),
        makePointerDecl(out_ptr.name, dt.elemType, tmp1),
        // create boolean flag used for swapping
        DeclStmt(VarDecl(flag.name, Type.uchar, Some(Literal("1")))),
        // generate body
        cmd(
          Phrase.substitute(Pair(Pair(Pair(ve, va), swap), done), `for` = ps, `in` = p),
          env updatedIdentEnv (ve -> in_ptr) updatedIdentEnv (va -> out_ptr)
            updatedCommEnv (swap -> {
            Block(immutable.Seq(
              ExprStmt(Assignment(in_ptr, TernaryExpr(flag, tmp1, tmp2))),
              ExprStmt(Assignment(out_ptr, TernaryExpr(flag, tmp2, tmp1))),
              // toggle flag with xor
              ExprStmt(Assignment(flag, BinaryExpr(flag, ^, Literal("1"))))))
          })
            updatedCommEnv (done -> {
            Block(immutable.Seq(
              ExprStmt(Assignment(in_ptr, TernaryExpr(flag, tmp1, tmp2))),
              acc(out, env, CIntExpr(0) :: Nil, o => ExprStmt(Assignment(out_ptr, UnaryExpr(&, o))))))
          }))
      ))
    }

    def codeGenNewRegRot(n: Nat,
                         dt: DataType,
                         registers: Identifier[VarType],
                         rotate: Identifier[CommandType],
                         body: Phrase[CommandType],
                         env: Environment): Stmt = {
      import C.AST._

      val re = Identifier(s"${registers.name}_e", registers.t.t1)
      val ra = Identifier(s"${registers.name}_a", registers.t.t2)
      val rot = Identifier(s"${rotate.name}_rotate", rotate.t)

      val registerCount = n.eval // FIXME: this is a quick solution
      // TODO: variable array
      // val rs = (0 until registerCount).map(i => DeclRef(freshName(s"r${i}_"))).toArray

      val rs = DeclRef(freshName(s"rs_"))
      val rst = DPIA.Types.ArrayType(n, dt)

      Block(
        // rs.map(r => DeclStmt(VarDecl(r.name, typ(dt))))
        Array(DeclStmt(VarDecl(rs.name, typ(rst))))
          :+ cmd(
          Phrase.substitute(immutable.Map(registers -> Pair(re, ra), rotate -> rot), `in` = body),
          env updatedIdentEnv (re -> rs) updatedIdentEnv (ra -> rs)
            updatedCommEnv (rot -> Block(
            // (1 until registerCount).map(i => Assignment(rs(i-1), rs(i)))
            (1 until registerCount).map(i => ExprStmt(Assignment(generateAccess(rst, rs, CIntExpr(i - 1) :: Nil, env), generateAccess(rst, rs, CIntExpr(i) :: Nil, env))))
          ))
        )
      )
    }

    def codeGenFor(n: Nat,
                   i: Identifier[ExpType],
                   p: Phrase[CommandType],
                   unroll:Boolean,
                   env: Environment): Stmt = {
      val cI = C.AST.DeclRef(freshName("i_"))
      val range = RangeAdd(0, n, 1)
      val updatedGen = updatedRanges(cI.name, range)

       applySubstitutions(n, env.identEnv) |> (n => {

      range.numVals match {
        // iteration count is 0 => skip body; no code to be emitted
        case Cst(0) => C.AST.Comment("iteration count is 0, no loop emitted")

        // iteration count is 1 => no loop
        case Cst(1) =>
          C.AST.Stmts(C.AST.Stmts(
            C.AST.Comment("iteration count is exactly 1, no loop emitted"),
            C.AST.DeclStmt(C.AST.VarDecl(cI.name, C.AST.Type.int, init = Some(C.AST.ArithmeticExpr(0))))),
            updatedGen.cmd(p, env updatedIdentEnv (i -> cI)))

        case _ =>

          if(unroll) {
            val statements = for(index <- rangeAddToScalaRange(range)) yield {
              val indexPhrase = AsIndex(n, Natural(index))
              val newPhrase = Phrase.substitute(phrase=indexPhrase,`for`=i, in=p)
              immutable.Seq(updatedGen.cmd(newPhrase, env))
            }
            C.AST.Block(
              C.AST.Comment(s"Unrolling from ${range.start} until ${range.stop} by increments of ${range.step}") +:
                statements.flatten
            )
          } else {
            // default case
            val init = C.AST.VarDecl(cI.name, C.AST.Type.int, init = Some(C.AST.ArithmeticExpr(0)))
            val cond = C.AST.BinaryExpr(cI, C.AST.BinaryOperator.<, C.AST.ArithmeticExpr(n))
            val increment = C.AST.Assignment(cI, C.AST.ArithmeticExpr(NamedVar(cI.name, range) + 1))

            C.AST.ForLoop(C.AST.DeclStmt(init), cond, increment,
              C.AST.Block(immutable.Seq(updatedGen.cmd(p, env updatedIdentEnv (i -> cI)))))
          }
      }})
    }

    def codeGenForNat(n: Nat,
                      i: NatIdentifier,
                      p: Phrase[CommandType],
                      unroll:Boolean,
                      env: Environment): Stmt = {
      val cI = C.AST.DeclRef(freshName("i_"))
      val range = RangeAdd(0, n, 1)
      val updatedGen = updatedRanges(cI.name, range)

      applySubstitutions(n, env.identEnv) |> (n => {

      range.numVals match {
        // iteration count is 0 => skip body; no code to be emitted
        case Cst(0) => C.AST.Comment("iteration count is 0, no loop emitted")

        // iteration count is 1 => no loop
        case Cst(1) =>
          C.AST.Stmts(C.AST.Stmts(
            C.AST.Comment("iteration count is exactly 1, no loop emitted"),
            C.AST.DeclStmt(C.AST.VarDecl(cI.name, C.AST.Type.int, init = Some(C.AST.ArithmeticExpr(0))))),
            updatedGen.cmd(p, env))

        case _ =>
          // default case
          val init = C.AST.VarDecl(cI.name, C.AST.Type.int, init = Some(C.AST.ArithmeticExpr(0)))
          val cond = C.AST.BinaryExpr(cI, C.AST.BinaryOperator.<, C.AST.ArithmeticExpr(n))
          val increment = C.AST.Assignment(cI, C.AST.ArithmeticExpr(NamedVar(cI.name, range) + 1))

          if(unroll) {
            val statements = for (index <- rangeAddToScalaRange(range)) yield {
              updatedGen.generateNatDependentBody(`for` = i, `phrase`=p, at = Cst(index), env).body
            }
            C.AST.Block(
              C.AST.Comment(s"Unrolling from ${range.start} until ${range.stop} by increments of ${range.step}") +:
              statements.flatten
            )
          } else {
            val body = updatedGen.generateNatDependentBody(`for` = i, `phrase` = p, at = NamedVar(cI.name, range), env)

            C.AST.ForLoop(C.AST.DeclStmt(init), cond, increment,
              body
            )
          }
      }})
    }


    private def rangeAddToScalaRange(range:RangeAdd) = {
      val (start,stop,step) = {
        try {
          (range.start.evalLong, range.stop.evalLong, range.step.evalLong)
        } catch {
          case _:NotEvaluableException => throw new Exception(s"Cannot evaluate range $range in loop unrolling")
        }
      }
      start until stop by step
    }

    def codeGenIdxAcc(i: Phrase[ExpType],
                      a: Phrase[AccType],
                      env: Environment,
                      ps: Path,
                      cont: Expr => Stmt): Stmt = {
      exp(i, env, Nil, {
        case C.AST.Literal(text) => acc(a, env, CIntExpr(Cst(text.toInt)) :: ps, cont)
        case C.AST.DeclRef(name) => acc(a, env, CIntExpr(NamedVar(name, ranges(name))) :: ps, cont)
        case C.AST.ArithmeticExpr(ae) => acc(a, env, ae :: ps, cont)
        case cExpr:C.AST.Expr =>
          val arithVar = NamedVar(freshName("tmpIdx"))
          acc(a, env, CIntExpr(arithVar) :: ps, generated => C.AST.Block(immutable.Seq(
            C.AST.DeclStmt(C.AST.VarDecl(arithVar.name, C.AST.Type.int, Some(cExpr))),
            cont(generated)
          )))
      })
    }

    def codeGenIdxVecAcc(i: Phrase[ExpType],
                         a: Phrase[AccType],
                         env: Environment,
                         ps: Path,
                         cont: Expr => Stmt): Stmt = {
      exp(i, env, Nil, i => {
        val idx: ArithExpr = i match {
          case C.AST.DeclRef(name) => NamedVar(name, ranges(name))
          case C.AST.ArithmeticExpr(ae) => ae
        }

        acc(a, env, CIntExpr(idx) :: ps, cont)
      })
    }

    def codeGenLiteral(d: OperationalSemantics.Data): Expr = {
      d match {
        case i: IndexData =>
          C.AST.ArithmeticExpr(i.n)
        case _: IntData | _: FloatData | _: DoubleData | _: BoolData =>
          C.AST.Literal(d.toString)
        case ArrayData(a) => d.dataType match {
          case ArrayType(n, st) =>
            a.head match {
              case IntData(0) | FloatData(0.0f) | BoolData(false)
                if a.distinct.length == 1 =>
                C.AST.Literal("(" + s"($st[$n]){" + a.head + "})")
              case _ =>
                C.AST.Literal("(" + s"($st[$n])" + a.mkString("{", ",", "}") + ")")
            }
          case _ => error("Expected scalar or array types")
        }
      }
    }

    def codeGenUnaryOp(op: Operators.Unary.Value, e: Expr): Expr = {
      C.AST.UnaryExpr(op, e)
    }

    def codeGenIdx(i: Phrase[ExpType],
                   e: Phrase[ExpType],
                   env: Environment,
                   ps: Path,
                   cont: Expr => Stmt): Stmt = {
      exp(i, env, Nil, {
        case C.AST.DeclRef(name) => exp(e, env, CIntExpr(NamedVar(name, ranges(name))) :: ps, cont)
        case C.AST.ArithmeticExpr(ae) => exp(e, env, CIntExpr(ae) :: ps, cont)
        case cExpr:C.AST.Expr =>
          val arithVar = NamedVar(freshName("tmpIdx"))
          exp(e, env, CIntExpr(arithVar) :: ps, generated => C.AST.Block(immutable.Seq(
            C.AST.DeclStmt(C.AST.VarDecl(arithVar.name, C.AST.Type.int, Some(cExpr))),
            cont(generated)
          )))
      })
    }

    def codeGenForeignFunction(funDecl: ForeignFunction.Declaration,
                               inTs: collection.Seq[DataType],
                               outT: DataType,
                               args: collection.Seq[Phrase[ExpType]],
                               env: Environment,
                               ps: Path,
                               cont: Expr => Stmt): Stmt =
    {
      funDecl.definition match {
        case Some(funDef) =>
          addDeclaration(
            C.AST.FunDecl(funDecl.name,
              returnType = typ(outT),
              params = (funDef.params zip inTs).map {
                case (name, dt) => C.AST.ParamDecl(name, typ(dt))
              },
              body = C.AST.Code(funDef.body)))
        case _ =>
      }

      codeGenForeignCall(funDecl.name, args, env, Nil, cont)
    }

    def codeGenForeignCall(name: String,
                           args: collection.Seq[Phrase[ExpType]],
                           env: Environment,
                           args_ps: Path,
                           cont: Expr => Stmt): Stmt =
    {
      def iter(args: collection.Seq[Phrase[ExpType]], res: VectorBuilder[Expr]): Stmt = {
        //noinspection VariablePatternShadow
        args match {
          case a +: args =>
            exp(a, env, args_ps, a => iter(args, res += a))
          case _ => cont(
            C.AST.FunCall(C.AST.DeclRef(name), res.result()))
        }
      }

      iter(args, new VectorBuilder())
    }

    def codeGenBinaryOp(op: Operators.Binary.Value,
                        e1: Expr,
                        e2: Expr): Expr = {
      C.AST.BinaryExpr(e1, op, e2)
    }

    def generateAccess(dt: DataType,
                       accuExpr: Expr,
                       path: Path,
                       env: Environment): Expr = {
      path match {
        case Nil => accuExpr
        case (xj: TupleAccess) :: ps =>
          val tuAccPos = xj match {
            case FstMember => "_fst"
            case SndMember => "_snd"
          }
          generateAccess(dt, C.AST.StructMemberAccess(accuExpr, C.AST.DeclRef(tuAccPos)), ps, env)
        case (i: CIntExpr) :: _ =>
          dt match {
            case _: VectorType =>
              val data = C.AST.StructMemberAccess(accuExpr, C.AST.DeclRef("data"))
              C.AST.ArraySubscript(data, C.AST.ArithmeticExpr(i))
            case at: ArrayType =>
              val (k, ps) = flattenArrayIndices(at, path)
              generateAccess(dt, C.AST.ArraySubscript(accuExpr, C.AST.ArithmeticExpr(k)), ps, env)

            case dat: DepArrayType =>
              val (k, ps) = flattenArrayIndices(dat, path)
              generateAccess(dt, C.AST.ArraySubscript(accuExpr, C.AST.ArithmeticExpr(k)), ps, env)
            case x => throw new Exception(s"Expected an ArrayType that is accessed by the index but found $x instead.")
          }
        case _ =>
          throw new Exception(s"Can't generate access for `$dt' with `${path.mkString("[", "::", "]")}'")
      }
    }

    def flattenArrayIndices(dt: DataType, path: Path): (Nat, Path) = {
      assert(dt.isInstanceOf[ArrayType] || dt.isInstanceOf[DepArrayType])

      val (indicesAsPathElements, rest) = path.splitAt(countArrayLayers(dt))
      indicesAsPathElements.foreach(i => assert(i.isInstanceOf[CIntExpr]))
      val indices = indicesAsPathElements.map(_.asInstanceOf[CIntExpr].num)
      assert(rest.isEmpty || !rest.head.isInstanceOf[CIntExpr])


      val subMap = buildSubMap(dt, indices)

      (ArithExpr.substitute(flattenIndices(dt, indices), subMap), rest)
    }

    def countArrayLayers(dataType: DataType):Int = {
      dataType match {
        case ArrayType(_, et) => 1 + countArrayLayers(et)
        case DepArrayType(_, NatDataTypeFunction(_ ,et)) => 1 + countArrayLayers(et)
        case _ => 0
      }
    }

    def flattenIndices(dataType: DataType, indicies:List[Nat]):Nat = {
      (dataType, indicies) match {
        case (array:ArrayType, index::rest) =>
          numberOfElementsUntil(array, index) + flattenIndices(array.elemType, rest)
        case (array:DepArrayType, index::rest) =>
          numberOfElementsUntil(array, index) + flattenIndices(array.elemFType.body, rest)
        case (_,  Nil) => 0
        case t => throw new Exception(s"This should not happen, pair $t")
      }
    }

    //Computes the total number of element in an array at a given offset
    def numberOfElementsUntil(dt:ArrayType, at:Nat):Nat = {
      DataType.getTotalNumberOfElements(dt.elemType)*at
    }

    def numberOfElementsUntil(dt:DepArrayType, at:Nat):Nat = {
      BigSum(from=0, upTo = at-1, `for`=dt.elemFType.x, DataType.getTotalNumberOfElements(dt.elemFType.body))
    }

    private def getIndexVariablesScopes(dt:DataType):List[Option[NatIdentifier]] = {
      dt match {
        case ArrayType(_ , et) => None::getIndexVariablesScopes(et)
        case DepArrayType(_, NatDataTypeFunction(i, et)) => Some(i)::getIndexVariablesScopes(et)
        case _ => Nil
      }
    }

    private def buildSubMap(dt: DataType,
                            indices: immutable.Seq[Nat]): Predef.Map[Nat, Nat]  = {
      val bindings = getIndexVariablesScopes(dt)
      bindings.zip(indices).map({
        case (Some(binder), index) => Some((binder, index))
        case _ => None
      }).filter(_.isDefined).map(_.get).toMap[Nat, Nat]
    }

    implicit def convertBinaryOp(op: idealised.SurfaceLanguage.Operators.Binary.Value): idealised.C.AST.BinaryOperator.Value = {
      import idealised.SurfaceLanguage.Operators.Binary._
      op match {
        case ADD => C.AST.BinaryOperator.+
        case SUB => C.AST.BinaryOperator.-
        case MUL => C.AST.BinaryOperator.*
        case DIV => C.AST.BinaryOperator./
        case MOD => C.AST.BinaryOperator.%
        case GT => C.AST.BinaryOperator.>
        case LT => C.AST.BinaryOperator.<
        case EQ => C.AST.BinaryOperator.==
      }
    }

    implicit def convertUnaryOp(op: idealised.SurfaceLanguage.Operators.Unary.Value): idealised.C.AST.UnaryOperator.Value = {
      import idealised.SurfaceLanguage.Operators.Unary._
      op match {
        case NEG => C.AST.UnaryOperator.-
      }
    }

    def makePointerDecl(name: String,
                        elemType: DataType,
                        expr: Expr): Stmt = {
      import C.AST._
      DeclStmt(
        VarDecl(name, PointerType(typ(elemType)), Some(expr)))
    }
  }

  protected def applySubstitutions(n: Nat,
                                   identEnv: immutable.Map[Identifier[_ <: BasePhraseTypes], C.AST.DeclRef]): Nat = {
    // lift the substitutions from the Phrase level to the ArithExpr level
    val substitionMap = identEnv.filter(_._1.t match {
      case ExpType(IndexType(_)) => true
      case AccType(IndexType(_)) => true
      case _ => false
    }).map(i => (NamedVar(i._1.name), NamedVar(i._2.name))).toMap[ArithExpr, ArithExpr]
    ArithExpr.substitute(n, substitionMap)
  }


  private def visitAndGenerateNat[N <: C.AST.Node](node:N, env:Environment):N = {
    C.AST.Nodes.VisitAndRebuild(node, new C.AST.Nodes.VisitAndRebuild.Visitor() {
      override def post(n: Node): Node =
        n match {
          case C.AST.ArithmeticExpr(ae) => genNat(ae, env)
          case other => other
      }
    })
  }


  def genNat(n: Nat,
          env: Environment): Expr = {

    def boolExp(b:BoolExpr, env:Environment):C.AST.Expr = b match {
      case BoolExpr.True => C.AST.Literal("true")
      case BoolExpr.False => C.AST.Literal("false")
      case BoolExpr.ArithPredicate(lhs, rhs, op) =>
        val cOp = op match {
          case ArithPredicate.Operator.!= => C.AST.BinaryOperator.!=
          case ArithPredicate.Operator.== => C.AST.BinaryOperator.==
          case ArithPredicate.Operator.< => C.AST.BinaryOperator.<
          case ArithPredicate.Operator.<= => C.AST.BinaryOperator.<=
          case ArithPredicate.Operator.> => C.AST.BinaryOperator.>
          case ArithPredicate.Operator.>= => C.AST.BinaryOperator.>=
        }
        C.AST.BinaryExpr(C.AST.ArithmeticExpr(lhs), cOp, C.AST.ArithmeticExpr(rhs))
    }
    import C.AST
    n match {
      case Cst(c) => AST.Literal(c.toString)
      case Pow(b, ex) =>
        ex match {
          case Cst(2) => AST.BinaryExpr(genNat(b, env),AST.BinaryOperator.*, genNat(b, env))
          case _ => AST.Cast(AST.Type.int, AST.FunCall(AST.DeclRef("pow"), immutable.Seq(
            AST.Cast(AST.Type.float, genNat(b, env)), AST.Cast(AST.Type.float, genNat(b, env)))
          ))
        }
      case Log(b, x) => AST.Cast(AST.Type.int, AST.FunCall(AST.DeclRef("log" + b), immutable.Seq(
        AST.Cast(AST.Type.float, genNat(x, env)))
      ))
      case Prod(es) => es.foldLeft(AST.Literal("1"):AST.Expr)((accum:AST.Expr, e:ArithExpr) => {
        e match {
          case Pow(b, Cst(-1)) => C.AST.BinaryExpr(accum, AST.BinaryOperator./, genNat(b, env))
          case _ => C.AST.BinaryExpr(accum, AST.BinaryOperator.*, genNat(e, env))
        }
      })
      case Sum(es) => es.map(genNat(_, env)).reduceOption(AST.BinaryExpr(_, AST.BinaryOperator.+, _)).getOrElse(AST.Literal("0"))
      case Mod(a, n) => AST.BinaryExpr(genNat(a, env), AST.BinaryOperator.%, genNat(n, env))
      case v:Var => C.AST.DeclRef(v.toString)
      case IntDiv(n, d) => AST.BinaryExpr(genNat(n, env), AST.BinaryOperator./, genNat(d, env))
      case lu:Lookup => AST.FunCall(AST.DeclRef(s"lookup${lu.id}"), immutable.Seq(AST.Literal(lu.index.toString)))

      case lift.arithmetic.IfThenElse(cond, trueBranch, falseBranch) => AST.TernaryExpr(
        boolExp(cond, env), genNat(trueBranch, env), genNat(falseBranch, env))

      case natFunCall: NatFunCall =>
        if(natFunCall.args.isEmpty) {
          env.inlLetNatEnv(natFunCall.fun)
        } else {
          AST.FunCall(AST.DeclRef(natFunCall.name), natFunCall.args.map({
            case NatArg(n) => genNat(n, env)
            case LetNatIdArg(ident) => env.inlLetNatEnv(ident)
          }))
        }

      case sp: SteppedCase => genNat(sp.intoIfChain(), env)
      case otherwise => throw new Exception(s"Don't know how to print $otherwise")
    }
  }


  protected def genPad(n: Nat, l: Nat, r: Nat,
                       left: Expr, right: Expr,
                       i: CIntExpr, ps: Path,
                       array: Phrase[ExpType],
                       env: Environment,
                       cont: Expr => Stmt): Stmt = {

    exp(array, env, CIntExpr(i - l) ::ps, arrayExpr => {

      def cOperator(op:ArithPredicate.Operator.Value):C.AST.BinaryOperator.Value = op match {
        case ArithPredicate.Operator.< => C.AST.BinaryOperator.<
        case ArithPredicate.Operator.> => C.AST.BinaryOperator.>
        case ArithPredicate.Operator.>= => C.AST.BinaryOperator.>=
        case _ => null
      }

      def genBranch(lhs:ArithExpr, rhs:ArithExpr, operator:ArithPredicate.Operator.Value, taken:Expr, notTaken:Expr):Expr = {
        import BoolExpr._
        arithPredicate(lhs, rhs, operator) match {
          case True => taken
          case False => notTaken
          case _ => C.AST.TernaryExpr(
            C.AST.BinaryExpr(C.AST.ArithmeticExpr(lhs), cOperator(operator), C.AST.ArithmeticExpr(rhs)),
            taken, notTaken)
        }
      }
      cont(genBranch(i, l, ArithPredicate.Operator.<, left, genBranch(i, l + n, ArithPredicate.Operator.<, arrayExpr, right)))
    })
  }
}


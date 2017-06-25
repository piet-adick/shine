package idealised.OpenCL.FunctionalPrimitives

import idealised.DPIA.Compilation.RewriteToImperative
import idealised.DPIA.DSL._
import idealised.DPIA.Phrases.VisitAndRebuild.Visitor
import idealised.DPIA.Phrases._
import idealised.DPIA.Semantics.OperationalSemantics.{Data, Store}
import idealised.DPIA.Types._
import idealised.DPIA._
import idealised.OpenCL.CodeGeneration.CodeGenerator
import idealised.OpenCL.CodeGeneration.CodeGenerator.Environment
import idealised.OpenCL.GeneratableExp
import opencl.generator.OpenCLAST.{Expression, FunctionCall}

import scala.language.reflectiveCalls
import scala.xml.Elem


final case class OpenCLFunction(name: String,
                                inTs: Seq[DataType],
                                outT: DataType,
                                args: Seq[Phrase[ExpType]])
  extends ExpPrimitive with GeneratableExp {

  override lazy val `type`: ExpType =
    (inTs zip args).foreach{
      case (inT, arg) => arg :: exp"[$inT]"
    } -> exp"[$outT]"

  override def visitAndRebuild(f: Visitor): Phrase[ExpType] = {
    OpenCLFunction(name, inTs.map(f(_)), f(outT), args.map(VisitAndRebuild(_, f)))
  }

  override def codeGenExp(env: Environment): Expression = {
    FunctionCall(name, args.map(CodeGenerator.exp(_, env)).toList)
  }

  override def eval(s: Store): Data = ???

  override def prettyPrint: String = s"$name(${args.map(PrettyPhrasePrinter(_)).mkString(",")})"

  override def xmlPrinter: Elem =
    <OpenCLFunction name={ToString(name)} inTs={ToString(inTs)} outT={ToString(outT)}>
      {args.map(Phrases.xmlPrinter(_))}
    </OpenCLFunction>

  override def acceptorTranslation(A: Phrase[AccType]): Phrase[CommandType] = {
    import RewriteToImperative._

    def recurse(ts: Seq[(Phrase[ExpType], DataType)],
                exps: Seq[Phrase[ExpType]],
                inTs: Seq[DataType]): Phrase[CommandType] = {
      ts match {
        // with only one argument left to process return the assignment of the OpenCLFunction call
        case Seq( (arg, inT) ) =>
          con(arg)(λ(exp"[$inT]")(e =>
            A :=|outT| OpenCLFunction(name, inTs :+ inT, outT, exps :+ e) ))
        // with a `tail` of arguments left, recurse
        case Seq( (arg, inT), tail@_* ) =>
          con(arg)(λ(exp"[$inT]")(e => recurse(tail, exps :+ e, inTs :+ inT) ))
      }
    }

    recurse(args zip inTs, Seq(), Seq())
  }

  override def continuationTranslation(C: Phrase[->[ExpType, CommandType]]): Phrase[CommandType] = {
    import RewriteToImperative._

    def recurse(ts: Seq[(Phrase[ExpType], DataType)],
                es: Seq[Phrase[ExpType]],
                inTs: Seq[DataType]): Phrase[CommandType] = {
      ts match {
        // with only one argument left to process continue with the OpenCLFunction call
        case Seq( (arg, inT) ) =>
          con(arg)(λ(exp"[$inT]")(e => C(OpenCLFunction(name, inTs :+ inT, outT, es :+ e)) ))
        // with a `tail` of arguments left, recurse
        case Seq( (arg, inT), tail@_* ) =>
          con(arg)(λ(exp"[$inT]")(e => recurse(tail, es :+ e, inTs :+ inT) ))
      }
    }

    recurse(args zip inTs, Seq(), Seq())
  }

}

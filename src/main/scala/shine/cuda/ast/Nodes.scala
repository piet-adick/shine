package shine.cuda.ast

import shine.C.AST.Nodes.{VisitAndGenerateStmt, VisitAndRebuild}
import shine.C.AST._
import shine.{C, cuda}

case class VarDecl(override val name: String,
                   override val t: Type,
                   addressSpace: cuda.AddressSpace,
                   override val init: Option[Expr] = None)
  extends C.AST.VarDecl(name, t, init)
{
  override def visitAndRebuild(v: VisitAndRebuild.Visitor): VarDecl =
    cuda.ast.VarDecl(name, v(t), addressSpace, init.map(VisitAndRebuild(_, v)))
}

case class Synchronize() extends Stmt {
  override def visitAndRebuild(v: VisitAndRebuild.Visitor): Synchronize = this

  override def visitAndGenerateStmt(v: VisitAndGenerateStmt.Visitor): Stmt =
    this
}

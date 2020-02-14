package shine.cuda.ast

import arithexpr.arithmetic.ArithExpr
import shine.C.AST._
import shine.DPIA.Types.AddressSpace
import shine.OpenCL
import shine.OpenCL.AST.KernelDecl
import shine.cuda.BuiltInAttribute

object Printer {
  def apply(n: Node): String = (new Printer).printNode(n)
}

class Printer extends shine.OpenCL.AST.Printer {
  override def printStmt(s: Stmt): Unit = s match {
    case _: Synchronize => print("__synchronize()")
    case _ => super.printStmt(s)
  }

  override def toString(e: ArithExpr): String = e match {
    case of: BuiltInAttribute => of.toString

    case _ => super.toString(e)
  }

  override def printKernelDecl(k: KernelDecl): Unit = {
    print("__global__")
    println("")
    print(s"void ${k.name}(")
    k.params.foreach(p => {
      printDecl(p)
      if (!p.eq(k.params.last)) print(", ")
    })
    print(")")

    printStmt(k.body)
  }

  override def printParamDecl(p: ParamDecl): Unit = {
    if (p.t.const) print("const ")
    p.t match {
      case b: BasicType => print(s"${b.name} ${p.name}")
      case s: StructType => print(s"struct ${s.name} ${p.name}")
      case _: UnionType => ???
      case _: ArrayType =>
        throw new Exception("Arrays as parameters are not supported")
      case pt: OpenCL.AST.PointerType =>
        val addrSpaceStr= toString(pt.a)
        val spaceAdj = if (addrSpaceStr.length == 0) "" else addrSpaceStr + " "
        print(s"$spaceAdj${pt.valueType}* __restrict__ ${p.name}")
      case _: shine.C.AST.PointerType =>
        throw new Exception(
          "Pointer without address space unsupported in OpenCL")
    }
  }

  override def toString(
    addressSpace: OpenCL.AddressSpace
  ): String = addressSpace match {
    case AddressSpace.Global  => ""
    case AddressSpace.Local   => "__shared__"
    case AddressSpace.Private => ""
    case _ => ???
  }
}

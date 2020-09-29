package shine.cuda.ast

case class ExternArrayType(override val elemType: shine.C.AST.Type) extends shine.C.AST.ArrayType(elemType, None, false) {
  override def print: String = "extern" + super.print
}

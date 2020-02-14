package shine.cuda

import shine.cuda

object KernelGenerator {
  def apply(): shine.OpenCL.KernelGenerator =
    new shine.OpenCL.KernelGenerator(
      cuda.codegen.CodeGenerator(),
      cuda.ast.Printer(_))
}

package shine.OpenCL

import arithexpr.arithmetic._
import opencl.executor._
import shine.C.AST.{Node}
import shine.DPIA.Phrases.Identifier
import shine.DPIA.Types._
import shine.DPIA._
import shine.{C, OpenCL}

import scala.collection.immutable.List
import scala.collection.Seq

//noinspection ScalaDocParserErrorInspection
case class Kernel(decls: Seq[C.AST.Decl],
                  kernel: OpenCL.AST.KernelDecl,
                  outputParam: Identifier[AccType],
                  inputParams: Seq[Identifier[ExpType]],
                  intermediateParams: Seq[Identifier[VarType]],
                  printer: Node => String
                  ) extends util.Kernel(decls, kernel, outputParam, inputParams, intermediateParams, printer) {

  override protected def execute(localSize: LocalSize, globalSize: GlobalSize, sizeVarMapping: Map[Nat, Nat], kernelArgs: List[KernelArg]) = {
    val kernelJNI = opencl.executor.Kernel.create(code, kernel.name, "")

    val kernelArgsOpenCL = kernelArgs.map(_.asInstanceOf[KernelArgOpenCL].kernelArg)

    val runtime = Executor.execute(kernelJNI,
      ArithExpr.substitute(localSize.size.x, sizeVarMapping).eval,
      ArithExpr.substitute(localSize.size.y, sizeVarMapping).eval,
      ArithExpr.substitute(localSize.size.z, sizeVarMapping).eval,
      ArithExpr.substitute(globalSize.size.x, sizeVarMapping).eval,
      ArithExpr.substitute(globalSize.size.y, sizeVarMapping).eval,
      ArithExpr.substitute(globalSize.size.z, sizeVarMapping).eval,
      kernelArgsOpenCL.toArray
    )

    runtime
  }

  override protected def findParameterMappings(arguments: List[Argument], localSize: LocalSize, globalSize: GlobalSize) : Map[Nat, Nat] = {
    val numGroups: NDRange = (
      globalSize.size.x /^ localSize.size.x,
      globalSize.size.y /^ localSize.size.y,
      globalSize.size.z /^ localSize.size.z)
    val sizeVarMapping = collectSizeVars(arguments, Map(
      get_num_groups(0) -> numGroups.x,
      get_num_groups(1) -> numGroups.y,
      get_num_groups(2) -> numGroups.z,
      get_local_size(0) -> localSize.size.x,
      get_local_size(1) -> localSize.size.y,
      get_local_size(2) -> localSize.size.z,
      get_global_size(0) -> globalSize.size.x,
      get_global_size(1) -> globalSize.size.y,
      get_global_size(2) -> globalSize.size.z
    ))

    sizeVarMapping
  }

  override protected def createLocalArg(sizeInByte: Long): KernelArg = {
    println(s"Allocated local argument with $sizeInByte bytes")
    KernelArgOpenCL(LocalArg.create(sizeInByte))
  }

  private def createGlobalArg(sizeInByte: Long): GlobalArg = {
    println(s"Allocated global argument with $sizeInByte bytes")
    GlobalArg.createOutput(sizeInByte)
  }

  private def createGlobalArg(array: Array[Float]): GlobalArg = {
    println(s"Allocated global argument with ${array.length * 4} bytes")
    GlobalArg.createInput(array)
  }

  private def createGlobalArg(array: Array[Int]): GlobalArg = {
    println(s"Allocated global argument with ${array.length * 4} bytes")
    GlobalArg.createInput(array)
  }

  private def createGlobalArg(array: Array[Double]): GlobalArg = {
    println(s"Allocated global argument with ${array.length * 8} bytes")
    GlobalArg.createInput(array)
  }

  private def createValueArg(value: Float): ValueArg = {
    println(s"Allocated value argument with 4 bytes")
    ValueArg.create(value)
  }

  private def createValueArg(value: Int): ValueArg = {
    println(s"Allocated value argument with 4 bytes")
    ValueArg.create(value)
  }

  private def createValueArg(value: Double): ValueArg = {
    println(s"Allocated value argument with 8 bytes")
    ValueArg.create(value)
  }

  override protected def createInputArg(arg: Any): KernelArg = {
    KernelArgOpenCL(arg match {
      case  f: Float => createValueArg(f)
      case af: Array[Float] => createGlobalArg(af)
      case af: Array[Array[Float]] => createGlobalArg(af.flatten)
      case af: Array[Array[Array[Float]]] => createGlobalArg(af.flatten.flatten)
      case af: Array[Array[Array[Array[Float]]]] => createGlobalArg(af.flatten.flatten.flatten)

      case  i: Int => createValueArg(i)
      case ai: Array[Int] => createGlobalArg(ai)
      case ai: Array[Array[Int]] => createGlobalArg(ai.flatten)
      case ai: Array[Array[Array[Int]]] => createGlobalArg(ai.flatten.flatten)
      case ai: Array[Array[Array[Array[Int]]]] => createGlobalArg(ai.flatten.flatten.flatten)

      case  d: Double => createValueArg(d)
      case ad: Array[Double] => createGlobalArg(ad)
      case ad: Array[Array[Double]] => createGlobalArg(ad.flatten)
      case ad: Array[Array[Array[Double]]] => createGlobalArg(ad.flatten.flatten)
      case ad: Array[Array[Array[Array[Double]]]] => createGlobalArg(ad.flatten.flatten.flatten)

      case p: Array[(_, _)] => p.head match {
          case (_: Int, _: Float) =>
            GlobalArg.createInput(flattenToArrayOfInts(p.asInstanceOf[Array[(Int, Float)]]))
          case _ => ???
        }
      case pp: Array[Array[(_, _)]] => pp.head.head match {
        case (_: Int, _: Float) =>
          GlobalArg.createInput(pp.flatMap(a => flattenToArrayOfInts(a.asInstanceOf[Array[(Int, Float)]])))
        case _ => ???
      }

      case _ => throw new IllegalArgumentException("Kernel argument is of unsupported type: " +
        arg.getClass.getName)
    })
  }

  override protected def asArray[R](dt: DataType, output: KernelArg): R = {
    val outputOpenCL = output.asInstanceOf[KernelArgOpenCL].kernelArg.asInstanceOf[GlobalArg]

    (dt match {
      case shine.DPIA.Types.int => outputOpenCL.asIntArray()
      case shine.DPIA.Types.f32 => outputOpenCL.asFloatArray()
      case shine.DPIA.Types.f64 => outputOpenCL.asDoubleArray()
      case _ => throw new IllegalArgumentException("Return type of the given lambda expression " +
        "not supported: " + dt.toString)
    }).asInstanceOf[R]
  }

  override protected def createOutputArg(numberOfElements: Int, dataType: DataType): KernelArg = {
    val sizeInByte = sizeInBytes(dataType)*numberOfElements
    println(s"Allocated global argument with $sizeInByte bytes")
    KernelArgOpenCL(createGlobalArg(sizeInByte))
  }

  private def sizeInBytes(dt: DataType): Long = {
    dt match {
      case shine.DPIA.Types.bool => 1L
      case shine.DPIA.Types.int | shine.DPIA.Types.NatType => 4L
      case shine.DPIA.Types.u8 | shine.DPIA.Types.i8 => 1L
      case shine.DPIA.Types.u16 | shine.DPIA.Types.i16 | shine.DPIA.Types.f16 => 2L
      case shine.DPIA.Types.u32 | shine.DPIA.Types.i32 | shine.DPIA.Types.f32 => 4L
      case shine.DPIA.Types.u64 | shine.DPIA.Types.i64 | shine.DPIA.Types.f64 => 8L
      case _ => throw new Exception("This should not happen")
    }
  }
}

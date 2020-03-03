package shine.cuda

import arithexpr.arithmetic.ArithExpr
import shine.C.AST.Node
import shine.DPIA.Phrases.Identifier
import shine.DPIA.Types._
import shine.DPIA.{Nat, VarType}
import shine.{C, OpenCL}
import shine.OpenCL.{GlobalSize, LocalSize, NDRange, get_global_size, get_local_size, get_num_groups}
import yacx.{ByteArg, DoubleArg, FloatArg, HalfArg, IntArg, LongArg, Program, ShortArg}

import scala.collection.Seq
import scala.collection.immutable.List

//noinspection ScalaDocParserErrorInspection
case class Kernel(decls: Seq[C.AST.Decl],
                  kernel: OpenCL.AST.KernelDecl,
                  outputParam: Identifier[AccType],
                  inputParams: Seq[Identifier[ExpType]],
                  intermediateParams: Seq[Identifier[VarType]],
                  printer: Node => String
                 ) extends util.Kernel(decls, kernel, outputParam, inputParams, intermediateParams, printer) {

  //TODO: how does this works for CUDA?
  override protected def findParameterMappings(arguments: List[Argument], localSize: LocalSize, globalSize: GlobalSize): Map[Nat, Nat] = {
    val numGroups: NDRange = (
      globalSize.size.x /^ localSize.size.x,
      globalSize.size.y /^ localSize.size.y,
      globalSize.size.z /^ localSize.size.z)
    val sizeVarMapping = collectSizeVars(arguments, Map(
      gridDim('x') -> numGroups.x,
      gridDim('y') -> numGroups.y,
      gridDim('z') -> numGroups.z,
      blockDim('x') -> localSize.size.x,
      blockDim('y') -> localSize.size.y,
      blockDim('z') -> localSize.size.z,
      globalDim('x') -> globalSize.size.x,
      globalDim('y') -> globalSize.size.y,
      globalDim('z') -> globalSize.size.z
    ))

    sizeVarMapping
  }

  override protected def execute(localSize: LocalSize, globalSize: GlobalSize, sizeVarMapping: Map[Nat, Nat], kernelArgs: List[KernelArg]): Double = {
    val kernel = Program.create(code, this.kernel.name).compile()

    val kernelArgsCUDA = kernelArgs.map(_.asInstanceOf[KernelArgCUDA].kernelArg)

    val runtime = kernel.launch(
              ArithExpr.substitute(numGroups.x, sizeVarMapping).eval,
              ArithExpr.substitute(numGroups.y, sizeVarMapping).eval,
              ArithExpr.substitute(numGroups.z, sizeVarMapping).eval,
              ArithExpr.substitute(localSize.size.x, sizeVarMapping).eval,
              ArithExpr.substitute(localSize.size.y, sizeVarMapping).eval,
              ArithExpr.substitute(localSize.size.z, sizeVarMapping).eval,
              kernelArgsCUDA.toArray: _*
    )

    runtime.getLaunch().asInstanceOf[Double]
  }

  //TODO
  //no implemented yet (reserve shared memory?)
  //not sure what a localArg is
  //in OpenCl-LocalArg there will be nothing up- or downloaded only the ith KernelArg
  //of the Kernel will be set as "cl::__local(sizeInByte)"
  override protected def createLocalArg(sizeInBytes: Long): KernelArgCUDA = ???

  override protected def createOutputArg(numberOfElements: Int, dataType: DataType): KernelArgCUDA = {
    KernelArgCUDA(dataType match {
      case shine.DPIA.Types.i8 =>
        println(s"Allocated global byte-argument with $numberOfElements bytes")
        ByteArg.createOutput(numberOfElements);
      case shine.DPIA.Types.i16 =>
        println(s"Allocated global short-argument with ${numberOfElements * 2} bytes")
        ShortArg.createOutput(numberOfElements);
      case shine.DPIA.Types.i32 | shine.DPIA.Types.int =>
        println(s"Allocated global int-argument with ${numberOfElements * 4} bytes")
        IntArg.createOutput(numberOfElements);
      case shine.DPIA.Types.i64 =>
        println(s"Allocated global long-argument with ${numberOfElements * 8} bytes")
        LongArg.createOutput(numberOfElements);
      case shine.DPIA.Types.f16 =>
        println(s"Allocated global half-argument with ${numberOfElements * 2} bytes")
        HalfArg.createOutput(numberOfElements);
      case shine.DPIA.Types.f32 =>
        println(s"Allocated global float-argument with ${numberOfElements * 4} bytes")
        FloatArg.createOutput(numberOfElements);
      case shine.DPIA.Types.f64 =>
        println(s"Allocated global double-argument with ${numberOfElements * 8} bytes")
        DoubleArg.createOutput(numberOfElements);
      case _ => throw new IllegalArgumentException("Argh Return type of the given lambda expression " +
        "not supported: " + dataType.toString)
    })
  }

  override protected def asArray[R](dt: DataType, output: KernelArg): R = {
    val outputCUDA = output.asInstanceOf[KernelArgCUDA].kernelArg

    (dt match {
      case shine.DPIA.Types.i8 => outputCUDA.asInstanceOf[ByteArg].asByteArray()
      case shine.DPIA.Types.i16 => outputCUDA.asInstanceOf[ShortArg].asShortArray()
      case shine.DPIA.Types.i32 | shine.DPIA.Types.int => outputCUDA.asInstanceOf[IntArg].asIntArray()
      case shine.DPIA.Types.i64 => outputCUDA.asInstanceOf[LongArg].asLongArray()
      case shine.DPIA.Types.f16 => outputCUDA.asInstanceOf[HalfArg].asFloatArray()
      case shine.DPIA.Types.f32 => outputCUDA.asInstanceOf[FloatArg].asFloatArray()
      case shine.DPIA.Types.f64 => outputCUDA.asInstanceOf[DoubleArg].asDoubleArray()
      case _ => throw new IllegalArgumentException("Return type of the given lambda expression " +
        "not supported: " + dt.toString)
    }).asInstanceOf[R]
  }

  override protected def createInputArg(arg: Any): KernelArgCUDA = {
    KernelArgCUDA(arg match {
      case  b: Byte => createValueArg(b)
      case ab: Array[Byte] => createArrayArg(ab)
      case ab: Array[Array[Byte]] => createArrayArg(ab.flatten)
      case ab: Array[Array[Array[Byte]]] => createArrayArg(ab.flatten.flatten)
      case ab: Array[Array[Array[Array[Byte]]]] => createArrayArg(ab.flatten.flatten.flatten)

      case  s: Short => createValueArg(s)
      case as: Array[Short] => createArrayArg(as)
      case as: Array[Array[Short]] => createArrayArg(as.flatten)
      case as: Array[Array[Array[Short]]] => createArrayArg(as.flatten.flatten)
      case as: Array[Array[Array[Array[Short]]]] => createArrayArg(as.flatten.flatten.flatten)

      case  i: Int => createValueArg(i)
      case ai: Array[Int] => createArrayArg(ai)
      case ai: Array[Array[Int]] => createArrayArg(ai.flatten)
      case ai: Array[Array[Array[Int]]] => createArrayArg(ai.flatten.flatten)
      case ai: Array[Array[Array[Array[Int]]]] => createArrayArg(ai.flatten.flatten.flatten)

      case  l: Long => createValueArg(l)
      case al: Array[Long] => createArrayArg(al)
      case al: Array[Array[Long]] => createArrayArg(al.flatten)
      case al: Array[Array[Array[Long]]] => createArrayArg(al.flatten.flatten)
      case al: Array[Array[Array[Array[Long]]]] => createArrayArg(al.flatten.flatten.flatten)

      //TODO use halfValues

      case  f: Float => createValueArg(f)
      case af: Array[Float] => createArrayArg(af)
      case af: Array[Array[Float]] => createArrayArg(af.flatten)
      case af: Array[Array[Array[Float]]] => createArrayArg(af.flatten.flatten)
      case af: Array[Array[Array[Array[Float]]]] => createArrayArg(af.flatten.flatten.flatten)

      case  d: Double => createValueArg(d)
      case ad: Array[Double] => createArrayArg(ad)
      case ad: Array[Array[Double]] => createArrayArg(ad.flatten)
      case ad: Array[Array[Array[Double]]] => createArrayArg(ad.flatten.flatten)
      case ad: Array[Array[Array[Array[Double]]]] => createArrayArg(ad.flatten.flatten.flatten)

      case p: Array[(_, _)] => p.head match {
        case (_: Int, _: Float) =>
          IntArg.create(flattenToArrayOfInts(p.asInstanceOf[Array[(Int, Float)]]):_*)
        case _ => ???
      }
      case pp: Array[Array[(_, _)]] => pp.head.head match {
        case (_: Int, _: Float) =>
          IntArg.create(pp.flatMap(a => flattenToArrayOfInts(a.asInstanceOf[Array[(Int, Float)]])):_*)
        case _ => ???
      }

      case _ => throw new IllegalArgumentException("Kernel argument is of unsupported type: " +
        arg.getClass.getName)
    })
  }

  private def createArrayArg(array: Array[Byte]): ByteArg = {
    println(s"Allocated global byte-argument with ${array.length * 1} bytes")
    ByteArg.create(array:_*)
  }

  private def createArrayArg(array: Array[Short]): ShortArg = {
    println(s"Allocated global short-argument with ${array.length * 2} bytes")
    ShortArg.create(array:_*)
  }

  private def createArrayArg(array: Array[Int]): IntArg = {
    println(s"Allocated global int-argument with ${array.length * 4} bytes")
    IntArg.create(array:_*)
  }

  private def createArrayArg(array: Array[Long]): LongArg = {
    println(s"Allocated global long-argument with ${array.length * 8} bytes")
    LongArg.create(array:_*)
  }

  // TODO: use this and createValueArgHalf. Not sure how a half-Array can be passed to this class
  //  private def createArrayArgHalf(array: Array[Float]): HalfArg = {
  //    println(s"Allocated global half-argument with ${array.length * 2} bytes")
  //    HalfArg.create(array:_*)
  //  }

  private def createArrayArg(array: Array[Float]): FloatArg = {
    println(s"Allocated global float-argument with ${array.length * 4} bytes")
    FloatArg.create(array:_*)
  }

  private def createArrayArg(array: Array[Double]): DoubleArg = {
    println(s"Allocated global double-argument with ${array.length * 8} bytes")
    DoubleArg.create(array:_*)
  }

  private def createValueArg(value: Byte): yacx.KernelArg = {
    println(s"Allocated value byte-argument with 1 bytes")
    ByteArg.createValue(value)
  }

  private def createValueArg(value: Short): yacx.KernelArg = {
    println(s"Allocated value short-argument with 2 bytes")
    ShortArg.createValue(value)
  }

  private def createValueArg(value: Int): yacx.KernelArg = {
    println(s"Allocated value int-argument with 4 bytes")
    IntArg.createValue(value)
  }

  private def createValueArg(value: Long): yacx.KernelArg = {
    println(s"Allocated value long-argument with 8 bytes")
    LongArg.createValue(value)
  }

  // TODO
  //  private def createValueArgHalf(value: Float): yacx.KernelArg = {
  //    println(s"Allocated value half-argument with 2 bytes")
  //    HalfArg.createValue(value)
  //  }

  private def createValueArg(value: Float): yacx.KernelArg = {
    println(s"Allocated value float-argument with 4 bytes")
    FloatArg.createValue(value)
  }

  private def createValueArg(value: Double): yacx.KernelArg = {
    println(s"Allocated value double-argument with 8 bytes")
    DoubleArg.createValue(value)
  }
}

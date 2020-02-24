package shine.cuda

import yacx.{ByteArg, DoubleArg, FloatArg, HalfArg, IntArg, KernelArg, LongArg, ShortArg}
import shine.DPIA.Types.{ArrayType, DataType, DataTypeIdentifier, DepArrayType, IndexType, NatToDataApply, NatToDataLambda, PairType, ScalarType, VectorType, int}


object KernelArgCreator {
  def createLocalArg(sizeInByte: Long): KernelArg = {
    println(s"Allocated local argument with $sizeInByte bytes")
    throw new Exception("Not implemented")
    //ByteArg.create(sizeInByte) //no implemented yet (reserve shared memory?)
    //not sure what a localArg is
    //in OpenCl-LocalArg there will be nothing up- or downloaded only the ith KernelArg
    //of the Kernel will be set as "cl::__local(sizeInByte)"
  }

  def createGlobalArg(numberOfElements: Int, dt : DataType): KernelArg = {
    (getOutputType(dt) match {
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
        "not supported: " + dt.toString)
    })
  }

  def createGlobalArg(array: Array[Byte]): ByteArg = {
    println(s"Allocated global byte-argument with ${array.length * 1} bytes")
    ByteArg.create(array:_*)
  }

  def createGlobalArg(array: Array[Short]): ShortArg = {
    println(s"Allocated global short-argument with ${array.length * 2} bytes")
    ShortArg.create(array:_*)
  }

  def createGlobalArg(array: Array[Int]): IntArg = {
    println(s"Allocated global int-argument with ${array.length * 4} bytes")
    IntArg.create(array:_*)
  }

  def createGlobalArg(array: Array[Long]): LongArg = {
    println(s"Allocated global long-argument with ${array.length * 8} bytes")
    LongArg.create(array:_*)
  }

  def createGlobalArgHalf(array: Array[Float]): HalfArg = {
    println(s"Allocated global half-argument with ${array.length * 2} bytes")
    HalfArg.create(array:_*)
  }

  def createGlobalArg(array: Array[Float]): FloatArg = {
    println(s"Allocated global float-argument with ${array.length * 4} bytes")
    FloatArg.create(array:_*)
  }

  def createGlobalArg(array: Array[Double]): DoubleArg = {
    println(s"Allocated global double-argument with ${array.length * 8} bytes")
    DoubleArg.create(array:_*)
  }

  def createValueArg(value: Byte): KernelArg = {
    println(s"Allocated value byte-argument with 1 bytes")
    ByteArg.createValue(value)
  }

  def createValueArg(value: Short): KernelArg = {
    println(s"Allocated value short-argument with 2 bytes")
    ShortArg.createValue(value)
  }

  def createValueArg(value: Int): KernelArg = {
    println(s"Allocated value int-argument with 4 bytes")
    IntArg.createValue(value)
  }

  def createValueArg(value: Long): KernelArg = {
    println(s"Allocated value long-argument with 8 bytes")
    LongArg.createValue(value)
  }

  def createValueArgHalf(value: Float): KernelArg = {
    println(s"Allocated value half-argument with 2 bytes")
    HalfArg.createValue(value)
  }

  def createValueArg(value: Float): KernelArg = {
    println(s"Allocated value float-argument with 4 bytes")
    FloatArg.createValue(value)
  }

  def createValueArg(value: Double): KernelArg = {
    println(s"Allocated value double-argument with 8 bytes")
    DoubleArg.createValue(value)
  }

  def createInputArgFromScalaValue(arg: Any): yacx.KernelArg = {
    arg match {
      //Not sure about the abbreviation in cases (exist case b:?):
      case  b: Byte => createValueArg(b)
      case ab: Array[Byte] => createGlobalArg(ab)
      case ab: Array[Array[Byte]] => createGlobalArg(ab.flatten)
      case ab: Array[Array[Array[Byte]]] => createGlobalArg(ab.flatten.flatten)
      case ab: Array[Array[Array[Array[Byte]]]] => createGlobalArg(ab.flatten.flatten.flatten)

      case  s: Short => createValueArg(s)
      case as: Array[Short] => createGlobalArg(as)
      case as: Array[Array[Short]] => createGlobalArg(as.flatten)
      case as: Array[Array[Array[Short]]] => createGlobalArg(as.flatten.flatten)
      case as: Array[Array[Array[Array[Short]]]] => createGlobalArg(as.flatten.flatten.flatten)

      case  i: Int => createValueArg(i)
      case ai: Array[Int] => createGlobalArg(ai)
      case ai: Array[Array[Int]] => createGlobalArg(ai.flatten)
      case ai: Array[Array[Array[Int]]] => createGlobalArg(ai.flatten.flatten)
      case ai: Array[Array[Array[Array[Int]]]] => createGlobalArg(ai.flatten.flatten.flatten)

      case  l: Long => createValueArg(l)
      case al: Array[Long] => createGlobalArg(al)
      case al: Array[Array[Long]] => createGlobalArg(al.flatten)
      case al: Array[Array[Array[Long]]] => createGlobalArg(al.flatten.flatten)
      case al: Array[Array[Array[Array[Long]]]] => createGlobalArg(al.flatten.flatten.flatten)

      case  f: Float => createValueArg(f)
      case af: Array[Float] => createGlobalArg(af)
      case af: Array[Array[Float]] => createGlobalArg(af.flatten)
      case af: Array[Array[Array[Float]]] => createGlobalArg(af.flatten.flatten)
      case af: Array[Array[Array[Array[Float]]]] => createGlobalArg(af.flatten.flatten.flatten)

      case  d: Double => createValueArg(d)
      case ad: Array[Double] => createGlobalArg(ad)
      case ad: Array[Array[Double]] => createGlobalArg(ad.flatten)
      case ad: Array[Array[Array[Double]]] => createGlobalArg(ad.flatten.flatten)
      case ad: Array[Array[Array[Array[Double]]]] => createGlobalArg(ad.flatten.flatten.flatten)

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
    }
  }

  def castToOutputType[R](dt: DataType, output: KernelArg): R = {
    assert(dt.isInstanceOf[ArrayType] || dt.isInstanceOf[DepArrayType])
    (getOutputType(dt) match {
      case shine.DPIA.Types.i8 => output.asInstanceOf[ByteArg].asByteArray()
      case shine.DPIA.Types.i16 => output.asInstanceOf[ShortArg].asShortArray()
      case shine.DPIA.Types.i32 | shine.DPIA.Types.int => output.asInstanceOf[IntArg].asIntArray()
      case shine.DPIA.Types.i64 => output.asInstanceOf[LongArg].asLongArray()
      case shine.DPIA.Types.f16 => output.asInstanceOf[HalfArg].asFloatArray()
      case shine.DPIA.Types.f32 => output.asInstanceOf[FloatArg].asFloatArray()
      case shine.DPIA.Types.f64 => output.asInstanceOf[DoubleArg].asDoubleArray()
      case _ => throw new IllegalArgumentException("Return type of the given lambda expression " +
        "not supported: " + dt.toString)
    }).asInstanceOf[R]
  }

  // TODO: move these to util or something and make public, since both opencl and cuda use them??
  private def flattenToArrayOfInts(a: Array[(Int, Float)]): Array[Int] = {
    a.flatMap{ case (x,y) => Iterable(x, java.lang.Float.floatToIntBits(y)) }
  }

  private def getOutputType(dt: DataType): DataType = dt match {
    case _: ScalarType => dt
    case _: IndexType => int
    case _: DataTypeIdentifier => dt
    case VectorType(_, elem) => elem
    case PairType(fst, snd) =>
      val fstO = getOutputType(fst)
      val sndO = getOutputType(snd)
      if (fstO != sndO) {
        throw new IllegalArgumentException("no supported output type " +
          s"for heterogeneous pair: ${dt}")
      }
      fstO
    case ArrayType(_, elemType) => getOutputType(elemType)
    case DepArrayType(_, NatToDataLambda(_, elemType)) =>
      getOutputType(elemType)
    case DepArrayType(_, _) | _: NatToDataApply =>
      throw new Exception("This should not happen")
  }
}

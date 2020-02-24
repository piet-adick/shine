package shine.cuda

import arithexpr.arithmetic.{ArithExpr, Cst}
import shine.C.AST.{Node, ParamDecl}
import shine.DPIA.Phrases.Identifier
import shine.DPIA.Types._
import shine.DPIA.{Nat, NatIdentifier, VarType}
import shine.{C, OpenCL}
import shine.OpenCL.{FunctionHelper, GlobalSize, HList, LocalSize, NDRange, get_global_size, get_local_size, get_num_groups}
import util.{Time, TimeSpan}
import shine.cuda.KernelArgCreator._
import yacx.{Executor, Program, KernelArg}


import scala.language.implicitConversions
import scala.collection.Seq
import scala.collection.immutable.List
import scala.util.{Failure, Success, Try}

//noinspection ScalaDocParserErrorInspection
case class Kernel(decls: Seq[C.AST.Decl],
                  kernel: OpenCL.AST.KernelDecl,
                  outputParam: Identifier[AccType],
                  inputParams: Seq[Identifier[ExpType]],
                  intermediateParams: Seq[Identifier[VarType]],
                  printer: Node => String
                 ) {

  def code: String = decls.map(printer(_)).mkString("\n") +
    "\n\n" +
    printer(kernel)

  /** This method will return a Scala function which executed the kernel via cuda and returns its
   * // result and the time it took to execute the kernel.
   * //
   * // A type annotation `F` has to be provided which specifies the type of the Scala function.
   * // The following syntax is used for this (which implementation can be found in
   * // cuda/Types.scala):
   * //
   * // <kernel>.as[ScalaFunction `(` <Arg1Type> `,` <Arg2Type> `,` <...> `)=>` <ReturnType> ]
   * //
   * // where all text in < > is to be replaced with the appropriate Scala terms.
   * // An example for the dot product which takes two arrays of floats and returns an array of float
   * // would be:
   * //
   * // dotKernel.as[ScalaFunction `(` Array[Float] `,` Array[Float] `)=>` Array[Float]]
   * *
   * NB: If the kernel takes a single argument, then the invocation must use this syntax
   * (Example with a kernel of type Array[Float] => Array[Float]
   * *
   * val kernelF = kernel.as[ScalaFunction`(`Array[Float]`)=>`Array[Float]]
   * val (result, time) = kernelF(xs `;`)
   */
  def as[F <: FunctionHelper](localSize: LocalSize, globalSize: GlobalSize)
                             (implicit ev: F#T <:< HList): F#T => (F#R, TimeSpan[Time.ms]) = {
    hArgs: F#T => {
      val args: List[Any] = hArgs.toList
      assert(inputParams.length == args.length)
      Executor.loadLibary()

      //Match up corresponding dpia parameter/intermediate parameter/scala argument (when present) in a covenience
      //structure
      val arguments = constructArguments(inputParams zip args, intermediateParams, this.kernel.params.tail)

      //First, we want to find all the parameter mappings
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

      val (outputArg, inputArgs) = createKernelArgs(arguments, sizeVarMapping)
      val kernelArgs = outputArg :: inputArgs

      val kernelJNI = Program.create(code, "foo").compile()

      List(localSize match {
        case LocalSize(x) if !x.isEvaluable => Some(s"CUDA local size is not evaluable (currently set to $x)")
        case _ => None
      }, globalSize match {
        case GlobalSize(x) if !x.isEvaluable => Some(s"CUDA local size is not evaluable (currently set to $x)")
        case _ => None
      }).filter(_.isDefined).map(_.get) match {
        case Nil =>
        case problems =>
          val errorMessage = "Cannot run kernel:\n" ++ problems.reduce(_ ++ _)
          throw new Exception(errorMessage)
      }

      val runtime = kernelJNI.launch(
        ArithExpr.substitute(localSize.size.x, sizeVarMapping).eval,
        ArithExpr.substitute(localSize.size.y, sizeVarMapping).eval,
        ArithExpr.substitute(localSize.size.z, sizeVarMapping).eval,
        ArithExpr.substitute(globalSize.size.x, sizeVarMapping).eval,
        ArithExpr.substitute(globalSize.size.y, sizeVarMapping).eval,
        ArithExpr.substitute(globalSize.size.z, sizeVarMapping).eval,
        kernelArgs.toArray: _*
      )

      val output = asArray[F#R](outputParam.`type`.dataType, outputArg)

      (output, TimeSpan.inMilliseconds(runtime.getTotal().asInstanceOf[Double]))
    }
  }

  /**
   * A helper class to group together the various bits of related information that make up a parameter
   *
   * @param identifier The identifier representing the parameter in the dpia source
   * @param parameter  The CUDA parameter as appearing in the generated kernel
   * @param argValue   For non-intermediate input parameters, this carries the scala value to pass in the executor
   */
  private case class Argument(identifier: Identifier[ExpType], parameter: ParamDecl, argValue: Option[Any])

  private def constructArguments(inputs: Seq[(Identifier[ExpType], Any)],
                                 intermediateParams: Seq[Identifier[VarType]],
                                 oclParams: Seq[ParamDecl]): List[Argument] = {
    // For each input ...
    inputs.headOption match {
      // ... if we have no more, we look at the intermediates ...
      case None => intermediateParams.headOption match {
        // ... if we have no more, we should also be out of ParamDecls
        case None =>
          assert(oclParams.isEmpty)
          Nil
        case Some(param) =>
          // We must have an CUDA parameter
          assert(oclParams.nonEmpty, "Not enough cuda parameters")
          Argument(Identifier(param.name, param.t.t1), oclParams.head, None) ::
            constructArguments(inputs, intermediateParams.tail, oclParams.tail)
      }
      case Some((param, arg)) =>
        // We must have an CUDA parameter
        assert(oclParams.nonEmpty, "Not enough cuda parameters")
        Argument(param, oclParams.head, Some(arg)) :: constructArguments(inputs.tail, intermediateParams, oclParams.tail)
    }
  }

  /**
   * From the dpia paramters and the scala arguments, returns a Map of identifiers to sizes for "size vars",
   * the output kernel argument, and all the input kernel arguments
   */
  private def createKernelArgs(arguments: Seq[Argument],
                               sizeVariables: Map[Nat, Nat]): (KernelArg, List[KernelArg]) = {
    println(sizeVariables)

    //Now create the output kernel arg
    println(s"Create output argument")
    val kernelOutput = createOutputKernelArg(sizeVariables)

    //Finally, generate the input kernel args
    println(s"Create input arguments")
    val kernelArguments = createInputKernelArgs(arguments, sizeVariables)

    (kernelOutput, kernelArguments)
  }

  /**
   * Iterates through all the input arguments, collecting the values of all int-typed input parameters, which may appear
   * in the sizes of the other arguments.
   */
  private def collectSizeVars(arguments: List[Argument], sizeVariables: Map[Nat, Nat]): Map[Nat, Nat] = {
    def recordSizeVariable(sizeVariables: Map[Nat, Nat], arg: Argument) = {
      arg.identifier.t match {
        case ExpType(shine.DPIA.Types.int, read) =>
          arg.argValue match {
            case Some(i: Int) =>
              sizeVariables + ((NatIdentifier(arg.identifier.name), Cst(i)))
            case Some(num) =>
              throw new Exception(s"Int value for kernel argument ${arg.identifier.name} expected but $num (of type ${num.getClass.getName} found")
            case None =>
              throw new Exception("Int kernel parameter needs a value")
          }
        case _ => sizeVariables
      }
    }

    arguments.foldLeft(sizeVariables)(recordSizeVariable)
  }

  /**
   * Generates the input kernel args
   *
   * @param arguments
   * @param sizeVariables
   * @return
   */
  private def createInputKernelArgs(arguments: Seq[Argument], sizeVariables: Map[Nat, Nat]): List[KernelArg] = {

    //Helper for the creation of intermdiate arguments
    def createIntermediateArg(arg: Argument, sizeVariables: Map[Nat, Nat]): yacx.KernelArg = {
      //Try to substitue away all the free variables
      val rawSize = sizeInElements(arg.identifier.t.dataType).value
      val cleanSize = ArithExpr.substitute(rawSize, sizeVariables)
      Try(cleanSize.evalInt) match {
        case Success(actualSize) =>
          arg.parameter.t match {
            case OpenCL.AST.PointerType(a, _, _) => a match {
              case AddressSpace.Private => throw new Exception("'Private memory' is an invalid memory for cuda parameter")
              case AddressSpace.Local => createLocalArg(actualSize)
              case AddressSpace.Global => createOutputArg(actualSize, arg.identifier.t.dataType)
              case AddressSpace.Constant => ???
              case AddressSpaceIdentifier(_) => throw new Exception("This shouldn't happen")
            }
            case _ => throw new Exception("This shouldn't happen")
          }
        case Failure(_) => throw new Exception(s"Could not evaluate $cleanSize")
      }
    }

    arguments match {
      case Nil => Nil
      case arg :: remainingArgs =>
        val kernelArg = arg.argValue match {
          //We have a scala value - this is an input argument
          case Some(scalaValue) => createInputArg(scalaValue)
          //No scala value - this is an intermediate argument
          case None => createIntermediateArg(arg, sizeVariables)
        }
        kernelArg :: createInputKernelArgs(remainingArgs, sizeVariables)
    }
  }

  private def createOutputKernelArg(sizeVariables: Map[Nat, Nat]): yacx.KernelArg = {
    val rawSize = sizeInElements(this.outputParam.t.dataType).value
    val cleanSize = ArithExpr.substitute(rawSize, sizeVariables)
    Try(cleanSize.evalInt) match {
      case Success(actualSize) => createOutputArg(actualSize, this.outputParam.t.dataType)
      case Failure(_) => throw new Exception(s"Could not evaluate $cleanSize")
    }
  }
  
  private def sizeInElements(dt: DataType): SizeInElements = dt match {
    case v: VectorType =>  sizeInElements(v.elemType) * v.size
    case r: PairType => sizeInElements(r.fst) + sizeInElements(r.snd)
    case a: ArrayType => sizeInElements(a.elemType) * a.size
    case a: DepArrayType => ???
    case _ => SizeInElements(1)
  }

  case class SizeInElements(value: Nat) {
    def *(rhs: Nat) = SizeInElements(value * rhs)
    def +(rhs: SizeInElements) = SizeInElements(value + rhs.value)

    override def toString = s"$value elements"
  }
}

sealed case class KernelWithSizes(kernel: Kernel,
                                  localSize: LocalSize,
                                  globalSize: GlobalSize) {
  def as[F <: FunctionHelper](implicit ev: F#T <:< HList): F#T => (F#R, TimeSpan[Time.ms]) =
    kernel.as[F](localSize, globalSize)

  def code: String = kernel.code
}

sealed case class KernelNoSizes(kernel: Kernel) {
  //noinspection TypeAnnotation
  def as[F <: FunctionHelper](implicit ev: F#T <:< HList) = new {
    def apply(localSize: LocalSize, globalSize: GlobalSize): F#T => (F#R, TimeSpan[Time.ms]) =
      kernel.as[F](localSize, globalSize)

    def withSizes(localSize: LocalSize, globalSize: GlobalSize): F#T => (F#R, TimeSpan[Time.ms]) =
      kernel.as[F](localSize, globalSize)
  }

  def code: String = kernel.code
}

object Kernel {
  implicit def forgetSizes(k: KernelWithSizes): KernelNoSizes =
    KernelNoSizes(k.kernel)
}

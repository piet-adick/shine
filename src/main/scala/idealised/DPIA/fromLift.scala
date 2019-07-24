package idealised.DPIA

import idealised.DPIA.DSL._
import idealised.DPIA.Phrases._
import idealised.DPIA.Semantics.{OperationalSemantics => OpSem}
import idealised.DPIA.Types._
import idealised.SurfaceLanguage.Operators
import lift.core.{semantics => ls, types => lt}
import lift.{core => l}

object fromLift {
  def apply(expr: l.Expr): Phrase[_ <: PhraseType] = {
    if (!l.IsClosedForm(expr)) {
      throw new Exception(s"expression is not in closed form: $expr")
    }
    expression(expr)
  }

  def expression(expr: l.Expr): Phrase[_ <: PhraseType] = expr match {
    case l.TypedExpr(typedExpr, t) =>
      typedExpr match {
        case l.Identifier(name) =>
          Identifier(name, `type`(t))

        case l.Lambda(x, e) => t match {
          case lt.FunType(i, _) =>
            Lambda(Identifier(x.name, `type`(i)), expression(e))
          case _ => ???
        }
        case l.Apply(f, e) =>
          Lifting.liftFunction( // TODO: should we try to reduce by lifting here?
            expression(f).asInstanceOf[Phrase[FunType[PhraseType, PhraseType]]])
            .value(expression(e).asInstanceOf[Phrase[PhraseType]])

        case l.DepLambda(x, e) => x match {
          case n: l.NatIdentifier =>
            DepLambda[NatKind](natIdentifier(n))(expression(e))
          case dt: lt.DataTypeIdentifier =>
            DepLambda[DataKind](dataTypeIdentifier(dt))(expression(e))
        }
        case l.DepApply(f, x) => x match {
          case n: Nat =>
            DepApply[NatKind, PhraseType]( // TODO: should we try to reduce by lifting here?
              expression(f).asInstanceOf[Phrase[DepFunType[NatKind, PhraseType]]],
              n)
          case dt: lt.DataType =>
            DepApply[DataKind, PhraseType]( // TODO: should we try to reduce by lifting here?
              expression(f).asInstanceOf[Phrase[DepFunType[DataKind, PhraseType]]],
              dataType(dt)
            )
        }

        case l.Literal(d)   =>  Literal(data(d))
        case l.Index(n, sz) =>  Literal(OpSem.IndexData(n, IndexType(sz)))
        case l.NatExpr(n)   =>  Natural(n)
        case p: l.Primitive =>  primitive(p, t)

        case _: l.TypedExpr => ??? // do not expect typed expr
      }
    case _ => ??? // expected typed expr
    }

  def access(w: lt.AccessType): AccessType = w match {
    case lt.W => write
    case lt.R => read
    case lt.AccessTypeIdentifier(name) => AccessTypeIdentifier(name)
  }

  def addressSpace(a: lt.AddressSpace): AddressSpace = a match {
    case lt.AddressSpace.Global => AddressSpace.Global
    case lt.AddressSpace.Local => AddressSpace.Local
    case lt.AddressSpace.Private => AddressSpace.Private
  }

  def scalarType(t: lt.ScalarType): ScalarType = t match {
    case lt.bool => bool
    case lt.int => int
    case lt.float => float
    case lt.double => double
    case lt.NatType => NatType
  }

  def basicType(t: lt.BasicType): BasicType = t match {
    case st: lt.ScalarType => scalarType(st)
    case lt.IndexType(sz) => IndexType(sz)
    case lt.VectorType(sz, et) => et match {
      case e : lt.ScalarType => VectorType(sz, scalarType(e))
      case _ => ???
    }
  }

  def dataType(t: lt.DataType): DataType = t match {
    case bt: lt.BasicType => basicType(bt)
    case i: lt.DataTypeIdentifier => dataTypeIdentifier(i)
    case lt.ArrayType(sz, et) => ArrayType(sz, dataType(et))
    case lt.DepArrayType(sz, f) => DepArrayType(sz, ntd(f))
    case lt.TupleType(a, b) => RecordType(dataType(a), dataType(b))
    case lt.NatToDataApply(f, n) => NatToDataApply(ntd(f), n)
  }

  def ntd(ntd: lt.NatToData): NatToData= ntd match {
    case lt.NatToDataLambda(n, body) => NatToDataLambda(natIdentifier(n), dataType(body))
    case lt.NatToDataIdentifier(x) => NatToDataIdentifier(x)
  }

  def ntn(ntn: lt.NatToNat): NatToNat= ntn match {
    case lt.NatToNatLambda(n, body) => NatToNatLambda(natIdentifier(n), body)
    case lt.NatToNatIdentifier(x) => NatToNatIdentifier(x)
  }

  def dataTypeIdentifier(dt: lt.DataTypeIdentifier): DataTypeIdentifier = DataTypeIdentifier(dt.name)
  def natIdentifier(n: l.NatIdentifier): NatIdentifier = NatIdentifier(n.name, n.range)
  def natToNatIdentifier(n: lt.NatToNatIdentifier): NatToNatIdentifier = NatToNatIdentifier(n.name)
  def natToDataIdentifier(n: lt.NatToDataIdentifier): NatToDataIdentifier = NatToDataIdentifier(n.name)

  def `type`(ty: lt.Type): PhraseType = ty match {
    case lt.DataAccessType(dt, w) => ExpType(dataType(dt), access(w))
      case lt.FunType(i, o)     => `type`(i) ->: `type`(o)
    case lt.DepFunType(i, t)  => i match {
        case dt: lt.DataTypeIdentifier    => dataTypeIdentifier(dt)   `()->:` `type`(t)
        case n: l.NatIdentifier           => natIdentifier(n)         `()->:` `type`(t)
        case n2n: lt.NatToNatIdentifier   => natToNatIdentifier(n2n)  `()->:` `type`(t)
        case n2d: lt.NatToDataIdentifier  => natToDataIdentifier(n2d) `()->:` `type`(t)
      }
  }

  def data(d: ls.Data): OpSem.Data = d match {
    case ls.ArrayData(a) => OpSem.ArrayData(a.map(data).toVector)
    case ls.TupleData(a, b) => OpSem.RecordData(data(a), data(b))
    case ls.BoolData(b) => OpSem.BoolData(b)
    case ls.IntData(i) => OpSem.IntData(i)
    case ls.FloatData(f) => OpSem.FloatData(f)
    case ls.DoubleData(f) => OpSem.DoubleData(f)
    case ls.VectorData(v) => OpSem.VectorData(v.map(data(_)).toVector)
  }

  import idealised.DPIA.FunctionalPrimitives._
  import lift.core.{primitives => core}

  def fun[T <: PhraseType](t: T,
                           f: Phrase[T] => Phrase[_ <: PhraseType]): Phrase[_ <: PhraseType] = {
    val x = Identifier(freshName("x"), t)
    Lambda(x, f(x))
  }

  def primitive(p: l.Primitive, t: lt.Type): Phrase[_ <: PhraseType] = {
    import idealised.OpenCL.FunctionalPrimitives._
    import idealised.OpenCL.{GlobalMemory, LocalMemory, PrivateMemory}
    import idealised.OpenMP.FunctionalPrimitives._
    import lift.OpenCL.{primitives => ocl}
    import lift.OpenMP.{primitives => omp}

    // TODO: remove surface language

    (p, t) match {
      case (core.`natAsIndex`,
      lt.DepFunType(n: l.NatIdentifier,
      lt.FunType(lt.DataAccessType(lt.NatType, lt.R), lt.DataAccessType(lt.IndexType(_), lt.R))))
      =>
        DepLambda[NatKind](natIdentifier(n))(
          fun[ExpType](ExpType(NatType, read), e =>
            AsIndex(n, e)))

      case (core.map,
      lt.FunType(lt.FunType(_, lt.DataAccessType(lb: lt.DataType, lt.W)),
      lt.FunType(lt.DataAccessType(lt.ArrayType(n, la: lt.DataType), lt.R), _)))
      =>
        makeMap(Map, n, la, lb)

      case (core.mapSeq,
      lt.FunType(lt.FunType(_, lt.DataAccessType(lb: lt.DataType, lt.W)),
      lt.FunType(lt.DataAccessType(lt.ArrayType(n, la: lt.DataType), lt.R), _)))
      =>
        makeMap(MapSeq, n, la, lb)

      case (omp.mapPar,
      lt.FunType(lt.FunType(_, lt.DataAccessType(lb: lt.DataType, lt.W)),
      lt.FunType(lt.DataAccessType(lt.ArrayType(n, la: lt.DataType), lt.R), _)))
      =>
        makeMap(MapPar, n, la, lb)

      case (ocl.mapGlobal(dim),
      lt.FunType(lt.FunType(_, lt.DataAccessType(lb: lt.DataType, lt.W)),
      lt.FunType(lt.DataAccessType(lt.ArrayType(n, la: lt.DataType), lt.R), _)))
      =>
        makeMap(MapGlobal(dim), n, la, lb)

      case (ocl.mapLocal(dim),
      lt.FunType(lt.FunType(_, lt.DataAccessType(lb: lt.DataType, lt.W)),
      lt.FunType(lt.DataAccessType(lt.ArrayType(n, la: lt.DataType), lt.R), _)))
      =>
        makeMap(MapLocal(dim), n, la, lb)

      case (ocl.mapWorkGroup(dim),
      lt.FunType(lt.FunType(_, lt.DataAccessType(lb: lt.DataType, lt.W)),
      lt.FunType(lt.DataAccessType(lt.ArrayType(n, la: lt.DataType), lt.R), _)))
      =>
        makeMap(MapWorkGroup(dim), n, la, lb)

      case (core.depMapSeq,
      lt.FunType(
      lt.DepFunType(lk: l.NatIdentifier, lt.FunType(_, _)),
      lt.FunType(lt.DataAccessType(lt.DepArrayType(n, la), lt.R), lt.DataAccessType(lt.DepArrayType(_, lb), lt.R))))
      =>
        val a = ntd(la)
        val b = ntd(lb)
        val k = natIdentifier(lk)
        fun[`(nat)->:`[ExpType ->: ExpType]](
          k `()->:` (ExpType(a(k), read) ->: ExpType(b(k), read))
          , f =>
          fun[ExpType](ExpType(DepArrayType(n, a), read), e =>
            DepMapSeq(n, a, b, f, e)))

      case (core.reduceSeq,
      lt.FunType(_,
      lt.FunType(lt.DataAccessType(lb: lt.DataType, lt.W),
      lt.FunType(lt.DataAccessType(lt.ArrayType(n, la), lt.R), _ ))))
      =>
        val a = dataType(la)
        val b = dataType(lb)
        fun[ExpType ->: ExpType ->: ExpType](exp"[$a, $read]" ->: exp"[$b, $read]" ->: exp"[$b, $write]", f =>
          fun[ExpType](exp"[$b, $read]", i =>
            fun[ExpType](exp"[$n.$a, $read]", e =>
              ReduceSeq(n, a, b, f, i, e))))

      case (ocl.oclReduceSeq,
      lt.DepFunType(i: lt.AddressSpaceIdentifier,
      lt.FunType(_,
      lt.FunType(lt.DataAccessType(lb: lt.DataType, lt.W),
      lt.FunType(lt.DataAccessType(lt.ArrayType(n, la), lt.R), _)))))
      =>
        val a = dataType(la)
        val b = dataType(lb)
        val i_space = addressSpace(i)
        fun[ExpType ->: ExpType ->: ExpType](exp"[$a, $read]" ->: exp"[$b, $read]" ->: exp"[$b, $write]", f =>
          fun[ExpType](exp"[$b, $read]", i =>
            fun[ExpType](exp"[$n.$a, $read]", e =>
              OpenCLReduceSeq(n, i_space, a, b, f, i, e))))

      case (core.reduceSeqUnroll,
      lt.FunType(_,
      lt.FunType(lt.DataAccessType(lb: lt.DataType, lt.W),
      lt.FunType(lt.DataAccessType(lt.ArrayType(n, la), lt.R), _ ))))
      =>
        val a = dataType(la)
        val b = dataType(lb)
        fun[ExpType ->: ExpType ->: ExpType](exp"[$a, $read]" ->: exp"[$b, $read]" ->: exp"[$b, $write]", f =>
          fun[ExpType](exp"[$b, $read]", i =>
            fun[ExpType](exp"[$n.$a, $read]", e =>
              ReduceSeqUnroll(n, a, b, f, i, e))))

      case (core.scanSeq,
      lt.FunType(_,
      lt.FunType(lt.DataAccessType(lb: lt.DataType, lt.W),
      lt.FunType(lt.DataAccessType(lt.ArrayType(n, la), lt.R), _))))
      =>
        val a = dataType(la)
        val b = dataType(lb)
        fun[ExpType ->: ExpType ->: ExpType](exp"[$a, $read]" ->: exp"[$b, $read]" ->: exp"[$b, $write]", f =>
          fun[ExpType](exp"[$b, $write]", i =>
            fun[ExpType](exp"[$n.$a, $read]", e =>
              ScanSeq(n, a, b, f, i, e))))

      case (core.depJoin,
        lt.FunType(lt.DataAccessType(lt.DepArrayType(n, llenF), lt.R), lt.DataAccessType(lt.ArrayType(_, la), lt.R)))
        =>
        val a = dataType(la)
        val lenF: NatToNatLambda = ??? // fromLift(llenF)
        fun[ExpType](exp"[$n.${NatToDataLambda(n, (i:NatIdentifier) => ArrayType(lenF(i), a))}, $read]", e =>
          DepJoin(n, lenF, a, e))

      case (core.join,
      lt.FunType(lt.DataAccessType(lt.ArrayType(n, lt.ArrayType(m, la)), lw), _))
      =>
        val a = dataType(la)
        val w = access(lw)
        fun[ExpType](exp"[$n.$m.$a, $w]", e =>
          Join(n, m, w, a, e))

      case (core.split,
      lt.DepFunType(n: l.NatIdentifier,
      lt.FunType(lt.DataAccessType(lt.ArrayType(mn, la), lw), lt.DataAccessType(lt.ArrayType(m, _), _))))
      =>
        val a = dataType(la)
        val w = access(lw)
        DepLambda[NatKind](natIdentifier(n))(
          fun[ExpType](exp"[$mn.$a, $w]", e =>
            Split(n, m, w, a, e)))

      case (core.slide,
      lt.DepFunType(sz: l.NatIdentifier,
      lt.DepFunType(sp: l.NatIdentifier,
      lt.FunType(lt.DataAccessType(lt.ArrayType(insz, la), lt.R), lt.DataAccessType(lt.ArrayType(n, _), lt.R)))))
      =>
        val a = dataType(la)
        DepLambda[NatKind](natIdentifier(sz))(
          DepLambda[NatKind](natIdentifier(sp))(
            fun[ExpType](exp"[$insz.$a, $read]", e =>
              Slide(n, sz, sp, a, e))))

      case (core.slideSeq(rot),
      lt.DepFunType(sz: l.NatIdentifier,
      lt.DepFunType(sp: l.NatIdentifier,
      lt.FunType(lt.DataAccessType(lt.ArrayType(insz, la), lt.R), lt.DataAccessType(lt.ArrayType(n, _), lt.R)))))
      =>
        val a = dataType(la)
        DepLambda[NatKind](natIdentifier(sz))(
          DepLambda[NatKind](natIdentifier(sp))(
            fun[ExpType](exp"[$insz.$a, $read]", e =>
              SlideSeq(rot, n, sz, sp, a, e))))

      case (core.reorder,
      lt.FunType(_,
      lt.FunType(_,
      lt.FunType(lt.DataAccessType(lt.ArrayType(n, la), lt.R), _))))
      =>
        val a = dataType(la)
        fun[ExpType ->: ExpType](exp"[idx($n), $read]" ->: exp"[idx($n), $read]", idxF =>
          fun[ExpType ->: ExpType](exp"[idx($n), $read]" ->: exp"[idx($n), $read]", idxFinv =>
            fun[ExpType](exp"[$n.$a, $read]", e =>
              Reorder(n, a, idxF, idxFinv, e))))

      case (core.transpose,
      lt.FunType(lt.DataAccessType(lt.ArrayType(n, lt.ArrayType(m, la)), lt.R), _))
      =>
        val a = dataType(la)

        val transposeFunction =
          λ(ExpType(IndexType(n * m), read))(i => {
            mapIndexExpr(i, j => {
              val col = (j % n) * m
              val row = j / n
              row + col
            })
          })

        val transposeInverseFunction =
          λ(ExpType(IndexType(n * m), read))(i => {
            mapIndexExpr(i, j => {
              val col = (j % m) * n
              val row = j / m
              row + col
            })
          })

        fun[ExpType](exp"[$n.$m.$a, read]", e =>
          Split(n, m, read, a,
            Reorder(n * m, a, transposeFunction, transposeInverseFunction,
              Join(n, m, read, a, e))))

      case (core.take,
      lt.DepFunType(n: l.NatIdentifier,
      lt.FunType(lt.DataAccessType(lt.ArrayType(nm, la), lw), _)))
      =>
        val m = nm - n
        val a = dataType(la)
        val w = access(lw)
        DepLambda[NatKind](natIdentifier(n))(
          fun[ExpType](exp"[$nm.$a, $w]", e =>
            Take(n, m, w, a, e)))

      case (core.drop,
      lt.DepFunType(n: l.NatIdentifier,
      lt.FunType(lt.DataAccessType(lt.ArrayType(nm, la), lw), _)))
      =>
        val m = nm - n
        val a = dataType(la)
        val w = access(lw)
        DepLambda[NatKind](natIdentifier(n))(
          fun[ExpType](exp"[$nm.$a, $w]", e =>
            Drop(n, m, w, a, e)))

      case (core.padCst,
      lt.DepFunType(l: l.NatIdentifier,
      lt.DepFunType(r: l.NatIdentifier,
      lt.FunType(_,
      lt.FunType(lt.DataAccessType(lt.ArrayType(n, la), lt.R), _)))))
      =>
        val a = dataType(la)
        DepLambda[NatKind](natIdentifier(l))(
          DepLambda[NatKind](natIdentifier(r))(
            fun[ExpType](exp"[$a, $read]", cst =>
                fun[ExpType](exp"[$n.$a, $read]", e =>
                  Pad(n, l, r, a, cst, e)))))

      case (core.padClamp,
      lt.DepFunType(l: l.NatIdentifier,
      lt.DepFunType(r: l.NatIdentifier,
      lt.FunType(lt.DataAccessType(lt.ArrayType(n, la), lt.R), _))))
      =>
        val a = dataType(la)
        DepLambda[NatKind](natIdentifier(l))(
          DepLambda[NatKind](natIdentifier(r))(
              fun[ExpType](exp"[$n.$a, $read]", e =>
                PadClamp(n, l, r, a, e))))

      case (core.unzip,
      lt.FunType(
      lt.DataAccessType(lt.ArrayType(n, lt.TupleType(la, lb)), lt.R),
      lt.DataAccessType(lt.TupleType(lt.ArrayType(_, _), lt.ArrayType(_, _)), lt.R)))
      =>
        val a = dataType(la)
        val b = dataType(lb)
        fun[ExpType](exp"[$n.($a x $b), $read]", e =>
            Unzip(n, a, b, e))

      case (core.zip,
      lt.FunType(lt.DataAccessType(lt.ArrayType(n, la), lt.R),
      lt.FunType(lt.DataAccessType(lt.ArrayType(_, lb), lt.R), _)))
      =>
        val a = dataType(la)
        val b = dataType(lb)
        fun[ExpType](exp"[$n.$a, $read]", x =>
          fun[ExpType](exp"[$n.$b, $read]", y =>
            Zip(n, a, b, x, y)))

      case (core.fst,
      lt.FunType(lt.DataAccessType(lt.TupleType(la, lb), lt.R), _))
      =>
        val a = dataType(la)
        val b = dataType(lb)
        fun[ExpType](exp"[($a x $b), $read]", e => Fst(a, b, e))

      case (core.snd,
      lt.FunType(lt.DataAccessType(lt.TupleType(la, lb), lt.R), _))
      =>
        val a = dataType(la)
        val b = dataType(lb)
        fun[ExpType](exp"[($a x $b), $read]", e => Snd(a, b, e))

      case (core.pair,
      lt.FunType(lt.DataAccessType(la: lt.DataType, lt.R),
      lt.FunType(lt.DataAccessType(lb: lt.DataType, lt.R), _)))
      =>
        val a = dataType(la)
        val b = dataType(lb)
        fun[ExpType](exp"[$a, $read]", x =>
          fun[ExpType](exp"[$b, $read]", y =>
            Record(a, b, x, y)))

      case (core.idx,
      lt.FunType(_,
      lt.FunType(lt.DataAccessType(lt.ArrayType(n, la), lt.R), _)))
      =>
        val a = dataType(la)
        fun[ExpType](exp"[idx($n), $read]", i =>
          fun[ExpType](exp"[$n.$a, $read]", e =>
            FunctionalPrimitives.Idx(n, a, i, e)))

      case (core.select,
      lt.FunType(_,
      lt.FunType(lt.DataAccessType(la: lt.DataType, lt.R), _)))
      =>
        val a = dataType(la)
        fun[ExpType](ExpType(bool, read), c =>
          fun[ExpType](ExpType(a, read), tExpr =>
            fun[ExpType](ExpType(a, read), fExpr =>
              IfThenElse(c, tExpr, fExpr))))

      case (core.neg, lt.FunType(lt.DataAccessType(la: lt.DataType, lt.R), _)) =>
        val a = dataType(la)
        fun[ExpType](exp"[$a, $read]", e => UnaryOp(Operators.Unary.NEG, e))

      case (core.add, lt.FunType(lt.DataAccessType(la: lt.DataType, lt.R), _)) =>
        val a = dataType(la)
        fun[ExpType](exp"[$a, $read]", e1 =>
          fun[ExpType](exp"[$a, $read]", e2 => BinOp(Operators.Binary.ADD, e1, e2)))
      case (core.sub, lt.FunType(lt.DataAccessType(la: lt.DataType, lt.R), _)) =>
        val a = dataType(la)
        fun[ExpType](exp"[$a, $read]", e1 =>
          fun[ExpType](exp"[$a, $read]", e2 => BinOp(Operators.Binary.SUB, e1, e2)))
      case (core.mul, lt.FunType(lt.DataAccessType(la: lt.DataType, lt.R), _)) =>
        val a = dataType(la)
        fun[ExpType](exp"[$a, $read]", e1 =>
          fun[ExpType](exp"[$a, $read]", e2 => BinOp(Operators.Binary.MUL, e1, e2)))
      case (core.div, lt.FunType(lt.DataAccessType(la: lt.DataType, lt.R), _)) =>
        val a = dataType(la)
        fun[ExpType](exp"[$a, $read]", e1 =>
          fun[ExpType](exp"[$a, $read]", e2 => BinOp(Operators.Binary.DIV, e1, e2)))
      case (core.mod, lt.FunType(lt.DataAccessType(la: lt.DataType, lt.R), _)) =>
        val a = dataType(la)
        fun[ExpType](exp"[$a, $read]", e1 =>
          fun[ExpType](exp"[$a, $read]", e2 => BinOp(Operators.Binary.MOD, e1, e2)))

      case (core.gt, lt.FunType(lt.DataAccessType(la: lt.DataType, lt.R), _)) =>
        val a = dataType(la)
        fun[ExpType](exp"[$a, $read]", e1 =>
          fun[ExpType](exp"[$a, $read]", e2 => BinOp(Operators.Binary.GT, e1, e2)))
      case (core.lt, lt.FunType(lt.DataAccessType(la: lt.DataType, lt.R), _)) =>
        val a = dataType(la)
        fun[ExpType](exp"[$a, $read]", e1 =>
          fun[ExpType](exp"[$a, $read]", e2 => BinOp(Operators.Binary.LT, e1, e2)))
      case (core.equal, lt.FunType(lt.DataAccessType(la: lt.DataType, lt.R), _)) =>
        val a = dataType(la)
        fun[ExpType](exp"[$a, $read]", e1 =>
          fun[ExpType](exp"[$a, $read]", e2 => BinOp(Operators.Binary.EQ, e1, e2)))

      case (core.cast, lt.FunType(lt.DataAccessType(la: lt.BasicType, lt.R), lt.DataAccessType(lb: lt.BasicType, lt.R)))
      =>
        val a = basicType(la)
        val b = basicType(lb)
        fun[ExpType](ExpType(a, read), x =>
          Cast(a, b, x))

      case (l.ForeignFunction(decl, la), _)
      =>
        val (inTs, outT) = foreignFunIO(la)
        wrapForeignFun(decl, inTs, outT, Vector())

      case (core.generate, lt.FunType(_, lt.DataAccessType(lt.ArrayType(n, la), lt.R)))
      =>
        val a = dataType(la)
        fun[ExpType ->: ExpType](exp"[idx($n), $read]" ->: exp"[$a, $read]", f =>
          Generate(n, a, f))

      case (core.iterate,
      lt.DepFunType(k: l.NatIdentifier,
      lt.FunType(lt.DepFunType(ll: l.NatIdentifier,
      lt.FunType(lt.DataAccessType(lt.ArrayType(ln, _), lt.R), _)),
      lt.FunType(lt.DataAccessType(lt.ArrayType(insz, _), lt.R), lt.DataAccessType(lt.ArrayType(m, la), lt.R)))))
      =>
        val l = natIdentifier(ll)
        val n = ln /^ l
        val a = dataType(la)
        DepLambda[NatKind](natIdentifier(k))(
          fun[`(nat)->:`[ExpType ->: ExpType]](l `()->:` (exp"[$ln.$a, $read]" ->: exp"[$l.$a, $read]"), f =>
              fun[ExpType](exp"[$insz.$a, $read]", e =>
                Iterate(n, m, k, a, f, e))))

      case (core.asVector,
      lt.DepFunType(n: l.NatIdentifier,
      lt.FunType(lt.DataAccessType(lt.ArrayType(mn, la: lt.ScalarType), lt.R), lt.DataAccessType(lt.ArrayType(m, _), lt.R))))
      =>
        val a = scalarType(la)
        DepLambda[NatKind](natIdentifier(n))(
          fun[ExpType](exp"[$mn.$a, $read]", e =>
            AsVector(n, m, a, e)))

      case (core.asScalar, lt.FunType(lt.DataAccessType(lt.ArrayType(m, lt.VectorType(n, la: lt.ScalarType)), lt.R), _))
      =>
        val a = scalarType(la)
        fun[ExpType](ExpType(ArrayType(m, VectorType(n, a)), read), e =>
          AsScalar(m, n, a, e))

      case (core.vectorFromScalar, lt.FunType(_, lt.DataAccessType(lt.VectorType(n, la: lt.ScalarType), lt.R)))
      =>
        val a = scalarType(la)
        fun[ExpType](ExpType(a, read), e =>
          VectorFromScalar(n, a, e))

      case (core.indexAsNat, lt.FunType(lt.DataAccessType(lt.IndexType(n), lt.R), lt.DataAccessType(lt.NatType, lt.R)))
      =>
        fun[ExpType](exp"[idx($n), $read]", e =>
          IndexAsNat(n, e))

      case (ocl.`toMem`,
      lt.DepFunType(las: lt.AddressSpaceIdentifier,
      lt.FunType(lt.DataAccessType(la: lt.DataType, lt.W), _)))
      =>
        val a = dataType(la)
        val as = addressSpace(las)
        fun[ExpType](exp"[$a, $write]", e =>
          To(as, a, e))

      case (core.reduce, _) =>
        throw new Exception(s"$p has no implementation")
    }
  }

  private def makeMap(map: (Nat, DataType, DataType, Phrase[ExpType ->: ExpType], Phrase[ExpType]) => Phrase[_ <: PhraseType],
                      n: Nat,
                      la: lt.DataType,
                      lb: lt.DataType): Phrase[_ <: PhraseType] = {
    val a = dataType(la)
    val b = dataType(lb)
    fun[ExpType ->: ExpType](ExpType(a, read) ->: ExpType(b, write), f =>
      fun[ExpType](exp"[$n.$a, $read]", e =>
        map(n, a, b, f, e)))
  }

  def foreignFunIO(t: lt.Type): (Vector[DataType], DataType) = {
    t match {
      case lo: lt.DataType =>
        (Vector(), dataType(lo))
      case lt.FunType(laa, lb) => laa match {
        case lt.DataAccessType(la: lt.DataType, _) =>
          val a = dataType(la)
          val (i, o) = foreignFunIO(lb)
          (a +: i, o)
        case _ => ???
      }
      case lt.DepFunType(_, _) => throw new Exception("This should not be possible")
    }
  }

  def wrapForeignFun(decl: l.ForeignFunction.Decl,
                     intTs: Vector[DataType],
                     outT: DataType,
                     args: Vector[Phrase[ExpType]]): Phrase[_ <: PhraseType] = {
    val i = args.length
    if (i < intTs.length) {
      fun[ExpType](ExpType(intTs(i), read), a =>
        wrapForeignFun(decl, intTs, outT, args :+ a))
    } else {
      ForeignFunction(decl, intTs, outT, args)
    }
  }
}
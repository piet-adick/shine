package shine

import com.github.ghik.silencer.silent
import org.scalactic.source
import org.scalatest.{BeforeAndAfter, Tag}
import org.scalatest.matchers.should.Matchers
import org.scalatest.funsuite.AnyFunSuite
import util.{AssertSame, Time, TimeSpan}

package object test_util {
  @silent("define classes/objects inside of package objects")
  abstract class Tests extends AnyFunSuite with Matchers {
    protected def testCL(testName: String, testTags: Tag*)(testFun: => Any /* Assertion */)(implicit pos: source.Position): Unit = {
      opencl.executor.Executor.loadAndInit()

      test(testName, testTags:_*)(testFun)(pos)

      opencl.executor.Executor.shutdown()
    }

    protected def testCU(testName: String, testTags: Tag*)(testFun: => Any /* Assertion */)(implicit pos: source.Position): Unit = {
      yacx.Executor.loadLibary()

      test(testName, testTags:_*)(testFun)(pos)
    }

    private final def maxDifference = 0.0001f;

    protected def similar(f1: Float, f2: Float): Boolean = {
      Math.abs(f1-f2) < maxDifference
    }

    protected def similar(af1: Array[Float], af2: Array[Float]): Boolean = {
      af1 zip af2 map{pair => similar(pair._1, pair._2)} reduce(_&&_)
    }

    protected def similar(af1: Array[Array[Float]], af2: Array[Array[Float]]): Boolean = {
      af1 zip af2 map{pair => similar(pair._1, pair._2)} reduce(_&&_)
    }
  }

  @silent("define classes/objects inside of package objects")
  abstract class TestsWithExecutor extends Tests with BeforeAndAfter {
    before {
      opencl.executor.Executor.loadLibrary()
      opencl.executor.Executor.init()
    }

    after {
      opencl.executor.Executor.shutdown()
    }
  }

  @silent("define classes/objects inside of package objects")
  abstract class TestsWithYACX extends Tests with BeforeAndAfter {
    before {
      yacx.Executor.loadLibary()
    }
  }

  def runsWithSameResult[R, U <: Time.Unit](runs: Seq[(String, (R, TimeSpan[U]))])
                                           (implicit assertSame: AssertSame[R]): Unit = {
    runs.tail.foreach(r => assertSame(r._2._1, runs.head._2._1, s"${r._1} had a different result"))
    runs.foreach(r => println(s"${r._1} time: ${r._2._2}"))
  }
}

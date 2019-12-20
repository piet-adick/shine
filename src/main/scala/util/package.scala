import java.io.{File, PrintWriter}

package object util {
  def createTempFile(prefix: String, suffix: String): File = {
    val tmp = File.createTempFile(prefix, suffix)
    tmp.deleteOnExit()
    tmp
  }

  def writeToTempFile(prefix: String, suffix: String, content: String): File = {
    val tmp = createTempFile(prefix, suffix)
    new PrintWriter(tmp) {
      try {
        write(content)
      } finally {
        close()
      }
    }
    tmp
  }

  def readFile(path: String): String = {
    val source = io.Source.fromFile(path)
    try source.getLines.mkString("\n") finally source.close
  }

  def assertSame[T](a: T, b: T, msg: String)
                   (implicit same: AssertSame[T]): Unit = {
    same(a, b, msg)
  }

  def withExecutor[T](f: => T): T = {
    import opencl.executor._

    Executor.loadLibrary()
    Executor.init()
    try { f } finally { Executor.shutdown() }
  }
}
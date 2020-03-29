package shine.benchmark;

import java.io.IOException;

public class OpenCLBenchmarkCLBlastKeplerSGemmExe extends OpenCLBenchmarkCLBlastKeplerSGemm {
    public static void main(String[] args) throws IOException {
        //Load Libary
        opencl.executor.Executor.loadLibrary();

        //Init Executor
        opencl.executor.Executor.init();

        //Warmup + benchmark kernel + print with differnt dataSizes (dataSizes in OpenClBenchmarkUtils.java)
        OpenCLBenchmarkUtils.benchmark(kernel, options, creator);

        //Shutdown Executor
        opencl.executor.Executor.shutdown();
    }
}

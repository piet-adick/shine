import java.io.IOException;

public class OpenCLBenchmarkKeplerSGemmExe extends OpenCLBenchmarkKeplerSGemm {
    public static void main(String[] args) throws IOException {
        //Load Libary
        opencl.executor.Executor.loadLibrary();

        //Init Executor
        opencl.executor.Executor.init();

        //Warmup + benchmark kernel + print with differnt dataSizes (dataSizes in OpenClBenchmarkUtils.java)
        OpenCLBenchmarkUtils.benchmark(kernel, options, creator, dataSizes);

        //Shutdown Executor
        opencl.executor.Executor.shutdown();
    }
}


import opencl.executor.GlobalArg;
import opencl.executor.KernelArg;
import opencl.executor.ValueArg;

import java.io.IOException;

public class OpenCLBenchmarkKeplerBestSGemmTotal extends OpenCLBenchmarkKeplerBestSGemm {

    public static void main(String[] args) throws IOException {
        //Load Library
        opencl.executor.Executor.loadAndInit();

        //Warm up
        OpenCLBenchmarkUtilsTotal.benchmark(kernel, options, BenchmarkConfig.numberExecutionsWarmUp, creator, BenchmarkConfig.warmUpSize);

        OpenCLBenchmarkUtilsTotal.BenchmarkResult result = OpenCLBenchmarkUtilsTotal.benchmark(kernel, options, BenchmarkConfig.numberExecutions, creator, BenchmarkConfig.dataSizesGEMM);

        System.out.println(result);

        //Shutdown Executor
        opencl.executor.Executor.shutdown();
    }
}

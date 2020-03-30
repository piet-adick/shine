
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

        OpenCLBenchmarkUtilsTotal.BenchmarkResult result = OpenCLBenchmarkUtilsTotal.benchmark(kernel, options, BenchmarkConfig.numberExecutions, creator, dataSizes);

        String resultString = result.toString();
        for (long dataSize : dataSizes) {
            String empty = "";
            if (creator.getDataLength(dataSize) < 100)
                empty = "  ";
            resultString = resultString.replaceFirst("B: execution-time:", "B (" + creator.getDataLength(dataSize) + "x"
                    + creator.getDataLength(dataSize) + " matrices): execution-time:" + empty);
        }

        System.out.println(resultString);

        //Shutdown Executor
        opencl.executor.Executor.shutdown();
    }
}

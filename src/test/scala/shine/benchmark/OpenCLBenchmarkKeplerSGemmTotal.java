
import opencl.executor.GlobalArg;
import opencl.executor.KernelArg;
import opencl.executor.ValueArg;

import java.io.IOException;

public class OpenCLBenchmarkKeplerSGemmTotal extends OpenCLBenchmarkKeplerSGemm  {
    public static void main(String[] args) throws IOException {
        OpenCLBenchmarkUtilsTotal.run(kernel, options, creator, dataSizes);
    }
}

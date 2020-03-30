
import java.io.IOException;

public class OpenCLBenchmarkCLBlastKeplerSGemmTotal extends OpenCLBenchmarkCLBlastKeplerSGemm{
    public static void main(String[] args) throws IOException {
        OpenCLBenchmarkUtilsTotal.run(kernel, options, creator, dataSizes);
    }
}

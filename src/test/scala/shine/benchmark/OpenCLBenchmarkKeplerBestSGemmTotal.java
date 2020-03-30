
import java.io.IOException;

public class OpenCLBenchmarkKeplerBestSGemmTotal extends OpenCLBenchmarkKeplerBestSGemm {
    public static void main(String[] args) throws IOException {
        OpenCLBenchmarkUtilsTotal.run(kernel, options, creator, dataSizes);
    }
}

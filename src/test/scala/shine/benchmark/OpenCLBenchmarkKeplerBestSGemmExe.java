import java.io.IOException;

public class OpenCLBenchmarkKeplerBestSGemmExe extends OpenCLBenchmarkKeplerBestSGemm {
    public static void main(String[] args) throws IOException {
        //Warmup + benchmark kernel + print with differnt dataSizes (dataSizes in OpenClBenchmarkUtils.java)
        OpenCLBenchmarkUtils.benchmark(kernel, options, creator, dataSizes);
    }
}
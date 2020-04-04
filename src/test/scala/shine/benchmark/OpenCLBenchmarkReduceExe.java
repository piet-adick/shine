import java.io.IOException;

public class OpenCLBenchmarkReduceExe extends OpenCLBenchmarkReduce {
    public static void main(String[] args) throws IOException {
        //Warmup + benchmark kernel + print with differnt dataSizes (dataSizes in OpenClBenchmarkUtils.java)
        OpenCLBenchmarkUtilsReduce.benchmark(kernel, options, creator, dataSizes);
    }
}

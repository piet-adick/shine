
import java.io.IOException;

public class OpenCLBenchmarkReduceTotal extends OpenCLBenchmarkReduce  {
    public static void main(String[] args) throws IOException {
        OpenCLBenchmarkUtilsTotalReduce.run(kernel, options, creator, dataSizes);
    }
}

import java.io.IOException;

public class OpenCLBenchmarkVectorAddTotal extends OpenCLBenchmarkVectorAdd{
    public static void main(String[] args) throws IOException {
        OpenCLBenchmarkUtilsTotal.run(kernel, options, creator, dataSizes);
    }
}

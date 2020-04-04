import yacx.Executor;
import yacx.Executor.BenchmarkResult;
import yacx.IntArg;
import yacx.KernelArg;
import yacx.Options;

import java.io.IOException;

public class ExampleVectorAdd {

    public static void main(String[] args) throws IOException {
        Executor.loadLibrary();

        Executor.KernelArgCreator creator = new Executor.KernelArgCreator() {

            @Override
            public int getDataLength(long dataSizeBytes) {
                return (int) (dataSizeBytes/ IntArg.SIZE_BYTES);
            }

            @Override
            public int getGrid0(int dataLength) {
                return dataLength;
            }

            @Override
            public int getBlock0(int dataLength) {
                return 1;
            }

            @Override
            public KernelArg[] createArgs(int dataLength) {
                int[] a = new int[dataLength];
                int[] b = new int[dataLength];

                for (int i = 0; i < dataLength; i++) {
                    a[i] = i;
                    b[i] = 2*i;
                }

                return new KernelArg[] {IntArg.create(a), IntArg.create(b), IntArg.createOutput(dataLength)};
            }
        };

        //Warm up
        Executor.benchmark("vectorAdd", Options.createOptions(), BenchmarkConfig.numberExecutionsWarmUp, creator, BenchmarkConfig.warmUpSize);

        BenchmarkResult result =  Executor.benchmark("vectorAdd", Options.createOptions(), BenchmarkConfig.numberExecutions, creator, BenchmarkConfig.dataSizesVetorAdd);

        System.out.println(result);
    }
}

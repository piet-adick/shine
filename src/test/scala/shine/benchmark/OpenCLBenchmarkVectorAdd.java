
import opencl.executor.GlobalArg;
import opencl.executor.KernelArg;
import opencl.executor.ValueArg;

public class OpenCLBenchmarkVectorAdd {
    //Kernelname (kernelcode in file kernelname.cl)
    static String kernel = "vectorAdd";
    static String options = "";
    static long[] dataSizes = BenchmarkConfig.dataSizesVetorAdd;

    //KernelArgeCreator
    static KernelArgCreator creator = new KernelArgCreator() {

        @Override
        public int getDataLength(long dataSizeBytes) {
            return (int) (dataSizeBytes/ 4);
        }

        @Override
        public int getLocal0(int dataLength) {
            return 1;
        }

        @Override
        public int getGlobal0(int dataLength) {
            return dataLength;
        }

        @Override
        public KernelArg[] createArgs(int dataLength) {
            int[] a = new int[dataLength];
            int[] b = new int[dataLength];

            for (int i = 0; i < dataLength; i++) {
                a[i] = i;
                b[i] = 2*i;
            }

            return new KernelArg[] {GlobalArg.createInput(a), GlobalArg.createInput(b),
                    GlobalArg.createOutput(dataLength*4)};
        }
    };
}

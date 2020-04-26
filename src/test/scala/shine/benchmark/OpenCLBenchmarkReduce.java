
import opencl.executor.GlobalArg;
import opencl.executor.KernelArg;
import opencl.executor.ValueArg;


public class OpenCLBenchmarkReduce {
    //Kernelname (kernelcode in file kernelname.cl)
    static String kernel = "nvidiaDerived1";
    static String options = "";
    static long[] dataSizes = BenchmarkConfig.dataSizesReduceOpenCL;

    //KernelArgeCreator
    static KernelArgCreator creator = new KernelArgCreator(){
        @Override
        public int getDataLength(long dataSizeBytes) {
            return (int) (dataSizeBytes/4); //Float is 4 byte
        }

        @Override
        public KernelArg[] createArgs(int dataLength) {
            int inputSize = (dataLength >= 2048 * 128) ? dataLength : 2048 * 128;
            float[] in = new float[inputSize];
            for (int i = 0; i < in.length; i++) {
                in[i] = (i < dataLength) ? i : 0;
            }

            GlobalArg outputArg = GlobalArg.createOutput(inputSize * 4 / 2048);
            GlobalArg inputArg = GlobalArg.createInput(in);
            ValueArg nArg = ValueArg.create(inputSize);

            return new KernelArg[]{outputArg, inputArg, nArg};
        }

        @Override
        public int getLocal0(int dataLength) {
            return 128;
        }

        @Override
        public int getGlobal0(int dataLength) {
            return (dataLength >= 2048 * 128) ? dataLength : 2048 * 128;
        }
    };
}

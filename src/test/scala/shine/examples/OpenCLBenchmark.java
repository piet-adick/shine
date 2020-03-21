
import opencl.executor.GlobalArg;
import opencl.executor.KernelArg;
import opencl.executor.ValueArg;

import java.io.IOException;

public class OpenCLBenchmark {

    public static void main(String[] args) throws IOException {
        //Load Libary
        opencl.executor.Executor.loadLibrary();

        //Init Executor
        opencl.executor.Executor.init();

        //Kernelname (kernelcode in file kernelname.cl)
        String kernel = "dotProduct";
        String options = "";

        //KernelArgeCreator
        OpenCLBenchmarkUtils.KernelArgCreator creator = new OpenCLBenchmarkUtils.KernelArgCreator(){
            @Override
            public int getDataLength(long dataSizeBytes) {
                return (int) (dataSizeBytes/4); //Float is 4 byte
            }

            @Override
            public KernelArg[] createArgs(int dataLength) {
                float[] x = new float[dataLength];
				float[] y = new float[dataLength];

				for (int i = 0; i < dataLength; i++) {
					x[i] = i;
					y[i] = dataLength-i;
				}
				
				// for int-parameter n
                ValueArg nArg = ValueArg.create(dataLength);

                //for input-arrays
                GlobalArg xArg = GlobalArg.createInput(x);
                GlobalArg yArg = GlobalArg.createInput(y);

                //for outputarg
                //specify size in bytes!
                //Float is 4 bytes
                GlobalArg outputArg = GlobalArg.createOutput(dataLength*4);

                //Noch zwei andere komische Argumente wegen dem komischen Kernel
                GlobalArg x36 = GlobalArg.createOutput(4);
                GlobalArg x38 = GlobalArg.createOutput(dataLength*4);

                return new KernelArg[]{outputArg, nArg, xArg, yArg, x36, x38};
            }

            //Not sure about local and global size
            //It is not equal to gridDim and blockDim in CUDA
            @Override
            public int getLocal0(int dataLength) {
                return 1;
            }

            @Override
            public int getGlobal0(int dataLength) {
                return 1;
            }
        };

        //Warmup + benchmark kernel + print with differnt dataSizes (dataSizes in OpenClBenchmarkUtils.java)
        OpenCLBenchmarkUtils.benchmark(kernel, options, creator);

        //Shutdown Executor
        opencl.executor.Executor.shutdown();
    }
}

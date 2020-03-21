
import opencl.executor.GlobalArg;
import opencl.executor.KernelArg;
import opencl.executor.LocalArg;
import opencl.executor.ValueArg;

import java.io.IOException;

public class OpenCLBenchmarkReduce {

	final static int maxThreads = 64;
	final static int kernelNumber = 2; // 3 different reduce kernels available
	
	public static void main(String[] args) throws IOException {
		//Load Libary
        opencl.executor.Executor.loadLibrary();

        //Init Executor
        opencl.executor.Executor.init();

        //Kernelname (kernelcode in file kernelname.cl)
        String kernel = "reduce";
        String options = "";

        //KernelArgeCreator
        OpenCLBenchmarkUtilsReduce.KernelArgCreator creator = new OpenCLBenchmarkUtilsReduce.KernelArgCreator(){
            @Override
            public int getDataLength(long dataSizeBytes) {
				return (int) (dataSizeBytes/8); //Double is 8 byte
            }

            @Override
            public KernelArg[] createArgs(int dataLength) {
				double[] in = new double[dataLength];
				for (int i = 0; i < in.length; i++) {
					in[i] = i;
				}
				
				int blockSize = getLocal0(dataLength);
				int gridSize = getGridSize(dataLength);
				
				GlobalArg inputArg = GlobalArg.createInput(in);
				GlobalArg outputArg = GlobalArg.createOutput(gridSize * 8);
				ValueArg nArg = ValueArg.create(dataLength);
				ValueArg blockSizeArg = ValueArg.create(getLocal0(dataLength));
				LocalArg localArg = LocalArg.create(blockSize * 8);

                return new KernelArg[]{inputArg, outputArg, nArg, blockSizeArg, localArg};
            }

            @Override
            public int getLocal0(int dataLength) {
                return (dataLength < maxThreads*2) ? nextPow2((dataLength + 1)/ 2) : maxThreads;
            }

			public int getGridSize(int dataLength) {
				int threads = getLocal0(dataLength);
				return (dataLength + (threads * 2 - 1)) / (threads * 2);
			}

            @Override
            public int getGlobal0(int dataLength) {
				return getLocal0(dataLength) * getGridSize(dataLength);
            }
			
			public int nextPow2(int n) {
				int highestOneBit = Integer.highestOneBit(n);
				if (n == highestOneBit) {
					return n;
				}
				return highestOneBit << 1;
			}
			
        };

        //Warmup + benchmark kernel + print with differnt dataSizes (dataSizes in OpenClBenchmarkUtilsReduce.java)
        OpenCLBenchmarkUtilsReduce.benchmark(kernel, options, creator, kernelNumber);
		
        //Shutdown Executor
        opencl.executor.Executor.shutdown();
    }
}

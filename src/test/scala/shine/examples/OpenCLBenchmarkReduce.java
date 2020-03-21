
import opencl.executor.GlobalArg;
import opencl.executor.KernelArg;
import opencl.executor.LocalArg;
import opencl.executor.ValueArg;

import java.io.IOException;

public class OpenCLBenchmarkReduce {

	final static int maxThreads = 1024;
	final static int maxBlocks = 2147483647;

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
            public int getDataLength(double dataSizeBytes) {
				return (int) (dataSizeBytes/8); //Double is 8 byte
            }

            @Override
            public KernelArg[] createArgs(int dataLength) {
				double[] in = new double[dataLength];
				for (int i = 0; i < in.length; i++) {
					in[i] = i;
				}
				
				int blockSize = getLocal0(dataLength);
				int gridSize = getGlobal0(dataLength);
				
				GlobalArg inputArg = GlobalArg.createInput(in);
				GlobalArg outputArg = GlobalArg.createOutput(gridSize * 8);
				ValueArg nArg = ValueArg.create(dataLength);
				LocalArg localArg = LocalArg.create(blockSize * 8);

                return new KernelArg[]{inputArg, outputArg, nArg, localArg};
            }

            @Override
            public int getLocal0(int dataLength) {
                return maxThreads;
            }


            @Override
            public int getGlobal0(int dataLength) {
                int threads = getLocal0(dataLength);
				int blocks = (dataLength + (threads * 2 - 1)) / (threads * 2);
				return Math.min(blocks, maxBlocks);
            }
			
        };

        //Warmup + benchmark kernel + print with differnt dataSizes (dataSizes in OpenClBenchmarkUtilsReduce.java)
        OpenCLBenchmarkUtilsReduce.benchmark(kernel, options, creator);
		
        //Shutdown Executor
        opencl.executor.Executor.shutdown();
    }
}

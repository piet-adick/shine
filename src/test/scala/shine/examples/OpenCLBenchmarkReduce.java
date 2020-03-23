
import opencl.executor.GlobalArg;
import opencl.executor.KernelArg;
import opencl.executor.LocalArg;
import opencl.executor.ValueArg;

import java.io.IOException;

public class OpenCLBenchmarkReduce {
	
	public static void main(String[] args) throws IOException {
		//Load Libary
        opencl.executor.Executor.loadLibrary();

        //Init Executor
        opencl.executor.Executor.init();

        //Kernelname (kernelcode in file kernelname.cl)
        String kernel = "nvidiaDerived1";
        String options = "";

        //KernelArgeCreator
        OpenCLBenchmarkUtilsReduce.KernelArgCreator creator = new OpenCLBenchmarkUtilsReduce.KernelArgCreator(){
            @Override
            public int getDataLength(long dataSizeBytes) {
				return (int) (dataSizeBytes/4); //Float is 4 byte
            }

            @Override
            public KernelArg[] createArgs(int dataLength) {
				int inputSize = (dataLength >= 2048) ? dataLength : 2048;
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
				return (dataLength >= 2048) ? dataLength : 2048;
            }
			
        };

        //Warmup + benchmark kernel + print with differnt dataSizes (dataSizes in OpenClBenchmarkUtilsReduce.java)
        OpenCLBenchmarkUtilsReduce.benchmark(kernel, options, creator);
		
        //Shutdown Executor
        opencl.executor.Executor.shutdown();
    }
}

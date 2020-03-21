
import opencl.executor.GlobalArg;
import opencl.executor.KernelArg;
import opencl.executor.ValueArg;

import java.io.IOException;

public class OpenCLBenchmark2 {

    public static void main(String[] args) throws IOException {
        //Load Libary
        opencl.executor.Executor.loadLibrary();

        //Init Executor
        opencl.executor.Executor.init();

        //Kernelname (kernelcode in file kernelname.cl)
        String kernel = "keplerSgemm";
        String options = "";

        //KernelArgeCreator
        OpenCLBenchmarkUtils.KernelArgCreator creator = new OpenCLBenchmarkUtils.KernelArgCreator(){
            @Override
            public int getDataLength(long dataSizeBytes) {
				return (int) (dataSizeBytes/4); //Float is 4 byte
            }
			
			public int getDimension(int dataLength) {
				return (int) Math.sqrt(dataLength);
			}

            @Override
            public KernelArg[] createArgs(int dataLength) {
                int x = getDimension(dataLength);
				float alpha = 1f;
				float beta = 1f;
				float[] aMatrix = new float[dataLength];
				float[] bMatrix = new float[dataLength];
				float[] cMatrix = new float[dataLength];
				for (int i = 0; i < aMatrix.length; i++) {
					aMatrix[i] = i;
				}
				for (int i = 0; i < bMatrix.length; i++) {
					bMatrix[i] = i;
				}
				for (int i = 0; i < cMatrix.length; i++) {
					cMatrix[i] = i;
				}
				
				GlobalArg aArg = GlobalArg.createInput(aMatrix);
				GlobalArg bArg = GlobalArg.createInput(bMatrix);
				GlobalArg cArg = GlobalArg.createInput(cMatrix);
				GlobalArg outputArg = GlobalArg.createOutput(dataLength * 4);
				ValueArg alphaArg = ValueArg.create(alpha);
				ValueArg betaArg = ValueArg.create(beta);
				ValueArg xArg = ValueArg.create(x);

                return new KernelArg[]{aArg, bArg, cArg, alphaArg, betaArg, outputArg, xArg, xArg, xArg};
            }

            @Override
            public int getLocal0(int dataLength) {
                return 32;
            }
			
			@Override
            public int getLocal1(int dataLength) {
                return 8;
            }

            @Override
            public int getGlobal0(int dataLength) {
                return 256;
            }
			
			@Override
            public int getGlobal1(int dataLength) {
                return 128;
            }
        };

        //Warmup + benchmark kernel + print with differnt dataSizes (dataSizes in OpenClBenchmarkUtils.java)
        OpenCLBenchmarkUtils.benchmark(kernel, options, creator);
		
        //Shutdown Executor
        opencl.executor.Executor.shutdown();
    }
}
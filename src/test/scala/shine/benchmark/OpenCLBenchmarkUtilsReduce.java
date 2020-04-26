
import opencl.executor.Kernel;
import opencl.executor.KernelArg;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

public class OpenCLBenchmarkUtilsReduce {
    final static int numberExecutions = BenchmarkConfig.numberExecutions;

    public static void benchmark(String kernelName, String options, KernelArgCreator creator, long[] dataSizesBytes) throws IOException {
        opencl.executor.Executor.loadAndInit();

        if (dataSizesBytes == null)
            throw new NullPointerException();
        if (dataSizesBytes.length == 0)
            throw new IllegalArgumentException("not data sizes specificated");
        if (numberExecutions <= 0)
            throw new IllegalArgumentException("illegal number of executions: " + numberExecutions);
        if (options == null)
            throw new NullPointerException();

        // Absolute time Measurement
        long t0 = System.currentTimeMillis();

        // Create and compile the chosen Kernel
		Kernel kernelJNI = opencl.executor.Kernel.create(OpenCLBenchmarkUtils.loadFile("kernels/" + kernelName+".cl"), kernelName, options);

        // Array for result & total time
        double[] result = new double[dataSizesBytes.length];

        //Warm up
        int dataLength = creator.getDataLength(BenchmarkConfig.warmUpSize);
        KernelArg[] args = creator.createArgs(dataLength);

        opencl.executor.Executor.benchmark(kernelJNI, creator.getLocal0(dataLength), creator.getLocal1(dataLength), creator.getLocal2(dataLength),
                creator.getGlobal0(dataLength), creator.getGlobal1(dataLength), creator.getGlobal2(dataLength), args, BenchmarkConfig.numberExecutionsWarmUp, 0);

        for (KernelArg arg : args)
            arg.dispose();

        // Start run for every dataSize
        for (int i = 0; i < dataSizesBytes.length; i++) {
            long dataSize = dataSizesBytes[i];

            if (dataSize <= 0)
                throw new IllegalArgumentException();

            dataLength = creator.getDataLength(dataSize);
			int inputSize = (dataLength >= 2048 * 128) ? dataLength : 2048 * 128;

			int local0 = creator.getLocal0(dataLength);
			int local1 = creator.getLocal1(dataLength);
			int local2 = creator.getLocal2(dataLength);
			int global0 = creator.getGlobal0(dataLength);
			int global1 = creator.getGlobal1(dataLength);
			int global2 = creator.getGlobal2(dataLength);
			args = creator.createArgs(dataLength);
			
			// Simulate final array
			float[] out = new float[128];
			for (int j = 1; j <= out.length; j++) {
				int size = inputSize / 128;
				int now = j * size;
				int before = (j - 1) * size;
				float newSum = now * (now + 1) / 2;
				float oldSum = before * (before + 1) / 2;
				out[j - 1] = now * (now + 1) / 2 - before * (before + 1) / 2;
			}

            System.out.println("Benchmark: " + dataLength);

            double[] resultI = opencl.executor.Executor.benchmark(kernelJNI, local0, local1, local2, global0, global1, global2, 
																  args, numberExecutions, 0);
            
			// Prepare grid reduces (to reduce all the results of each block)
			dataLength /= 2048;
			
			while (dataLength > 128) {
				System.out.println("Repeating reduce for n = " + dataLength);
				local0 = creator.getLocal0(dataLength);
				global0 = creator.getGlobal0(dataLength);
				args = creator.createArgs(dataLength);
//                System.out.println("local0 = " + local0 + " global0 = " + global0);

				// Simulate grid reduce
				double[] resultI2 = opencl.executor.Executor.benchmark(kernelJNI, local0, 1, 1, global0, 1, 1, args, numberExecutions, 0);
				
				dataLength /= 2048;
				
				// Add average times
				for (int j = 0; j < numberExecutions; j++) {
					resultI[j] += resultI2[j];
				}
			}
			
			long dt1 = 0;
			
			// Simulate the final reduce on the CPU
			for (int j = 0; j < numberExecutions; j++) {
				float temp = 0;
				long t1 = System.currentTimeMillis();
				
				for (int k = 0; k < out.length; k++)
				temp += out[k];
			
				dt1 = dt1 + (System.currentTimeMillis() - t1);
			}
		
            for (int j = 0; j < numberExecutions; j++)
                result[i] += resultI[j];

            for (KernelArg arg : args)
                arg.dispose();

			result[i] += dt1;
            result[i] /= (double) numberExecutions;
        }

        // Absolute time Measurement
        long dt = System.currentTimeMillis() - t0;

        OpenCLBenchmarkUtils.BenchmarkResult r = new OpenCLBenchmarkUtils.BenchmarkResult(numberExecutions, dataSizesBytes, result, kernelName, dt);

        System.out.println(r);

        opencl.executor.Executor.shutdown();
    }
}

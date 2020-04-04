
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

			int local0 = creator.getLocal0(dataLength);
			int local1 = creator.getLocal1(dataLength);
			int local2 = creator.getLocal2(dataLength);
			int global0 = creator.getGlobal0(dataLength);
			int global1 = creator.getGlobal1(dataLength);
			int global2 = creator.getGlobal2(dataLength);
			args = creator.createArgs(dataLength);

//            System.out.println("Benchmark: " + dataLength);

            double[] resultI = opencl.executor.Executor.benchmark(kernelJNI, local0, local1, local2, global0, global1, global2, 
																  args, numberExecutions, 0);
            
			// Prepare grid reduces (to reduce all the results of each block)
			dataLength /= 2048;
			
			while (dataLength > 1) {
//				System.out.println("Repeating reduce for n = " + dataLength);
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

            for (int j = 0; j < numberExecutions; j++)
                result[i] += resultI[j];

            for (KernelArg arg : args)
                arg.dispose();

            result[i] /= (double) numberExecutions;
        }

        // Absolute time Measurement
        long dt = System.currentTimeMillis() - t0;

        OpenCLBenchmarkUtils.BenchmarkResult r = new OpenCLBenchmarkUtils.BenchmarkResult(numberExecutions, dataSizesBytes, result, kernelName, dt);

        System.out.println(r);

        opencl.executor.Executor.shutdown();
    }
}

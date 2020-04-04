
import opencl.executor.Kernel;
import opencl.executor.KernelArg;
import opencl.executor.KernelTime;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.text.DecimalFormat;

public class OpenCLBenchmarkUtilsTotalReduce {

    public static void run(String kernel, String options, KernelArgCreator creator, long[] dataSizes) throws IOException {
        //Load Library
        opencl.executor.Executor.loadAndInit();

        //Warm up
        OpenCLBenchmarkUtilsTotal.benchmark(kernel, options, BenchmarkConfig.numberExecutionsWarmUp, creator, BenchmarkConfig.warmUpSize);

        OpenCLBenchmarkUtilsTotal.BenchmarkResult result = OpenCLBenchmarkUtilsTotal.benchmark(kernel, options, BenchmarkConfig.numberExecutions, creator, dataSizes);

        System.out.println(result);

        //Shutdown Executor
        opencl.executor.Executor.shutdown();
    }

    public static OpenCLBenchmarkUtilsTotal.BenchmarkResult benchmark(String kernelName, String buildOptions, int numberExecutions, KernelArgCreator creator, long... dataSizesBytes) throws IOException {
        if (dataSizesBytes == null)
            throw new NullPointerException();
        if (dataSizesBytes.length == 0)
            throw new IllegalArgumentException("not data sizes specificated");
        if (numberExecutions <= 0)
            throw new IllegalArgumentException("illegal number of executions: " + numberExecutions);

        // Absolute time Measurement
        long t0 = System.currentTimeMillis();

        // Create and compile Kernel
        Kernel kernelJNI = opencl.executor.Kernel.create(OpenCLBenchmarkUtilsTotal.loadFile("kernels/" + kernelName+".cl"), kernelName, buildOptions);

        // Array for result
        KernelTime[][] result = new KernelTime[dataSizesBytes.length][numberExecutions];

        // Start run for every dataSize
        for (int i = 0; i < dataSizesBytes.length; i++) {
            long dataSize = dataSizesBytes[i];

            if (dataSize <= 0)
                throw new IllegalArgumentException();

            int dataLength = creator.getDataLength(dataSize);
            int local0 = creator.getLocal0(dataLength);
            int local1 = creator.getLocal1(dataLength);
            int local2 = creator.getLocal2(dataLength);
            int global0 = creator.getGlobal0(dataLength);
            int global1 = creator.getGlobal1(dataLength);
            int global2 = creator.getGlobal2(dataLength);
            KernelArg[] args = creator.createArgs(dataLength);

            // Execute Kernel numberExecutions times
            result[i] = opencl.executor.Executor.benchmark(kernelJNI, local0, local1, local2, global0, global1, global2,
                    args, numberExecutions, 0);

            // Prepare grid reduces (to reduce all the results of each block)
            dataLength /= 2048;

            while (dataLength > 1) {
                System.out.println("Repeating reduce for n = " + dataLength);
                local0 = creator.getLocal0(dataLength);
                global0 = creator.getGlobal0(dataLength);
                args = creator.createArgs(dataLength);

                // Simulate grid reduce
                KernelTime[] resultI = opencl.executor.Executor.benchmark(kernelJNI, local0, 1, 1, global0, 1, 1, args, numberExecutions, 0);

                dataLength /= 2048;

                // Add average times
                for (int j = 0; j < numberExecutions; j++) {
                    result[i][j] = result[i][j].addKernelTime(resultI[j]);
                }
            }

            // Destroy corresponding C-Objects
            for (KernelArg arg : args) {
                arg.dispose();
            }
        }

        // Absolute time Measurement
        long dt = System.currentTimeMillis() - t0;

        return new OpenCLBenchmarkUtilsTotal.BenchmarkResult(numberExecutions, dataSizesBytes, result, kernelName, dt);
    }
}

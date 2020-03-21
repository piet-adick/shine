
import opencl.executor.Kernel;
import opencl.executor.KernelArg;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

public class OpenCLBenchmarkUtilsReduce {
    final static long KB = 1024;
    final static long MB = 1024 * 1024;

    final static int numberExecutions = 10;

    final static long[] dataSizesBytes = new long[]{
			1 * KB,
			4 * KB,
			16 * KB,
			64 * KB,
			256 * KB,
			1 * MB,
			4 * MB,
            16 * MB,
			64 * MB};
			//256 * MB};
			//1024 * MB};
			
	public static String humanReadableByteCountBin(long bytes) {
            return bytes < 1024L ? bytes + " B"
                    : bytes <= 0xfffccccccccccccL >> 40 ? String.format("%.1f KiB", bytes / 0x1p10)
                    : bytes <= 0xfffccccccccccccL >> 30 ? String.format("%.1f MiB", bytes / 0x1p20)
                    : String.format("%.1f GiB", bytes / 0x1p30);
        }

    public static void benchmark(String kernelName, String options, KernelArgCreator creator) throws IOException {
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

        // Create and compile Kernels
        Kernel kernelJNI = opencl.executor.Kernel.create(loadFile(kernelName+".cl"), kernelName, options);
		Kernel kernelJNI2 = opencl.executor.Kernel.create(loadFile("reduce2.cl"), "reduce2", options);

        // Array for result & total time
        double[] result = new double[dataSizesBytes.length];
		double[] total = new double[dataSizesBytes.length];
		long totalStart;
		double total2;

		System.out.println("Warming up...");
		System.out.println("(Local = " + creator.getLocal0((int) (1 * MB)) + ", " + 
										 creator.getLocal1((int) (1 * MB)) + ", " + 
										 creator.getLocal2((int) (1 * MB)) + ")");
		System.out.println("(Global = " + creator.getGlobal0((int) (1 * MB)) + ", " + 
										 creator.getGlobal1((int) (1 * MB)) + ", " + 
										 creator.getGlobal2((int) (1 * MB)) + ")");

        //Warm up
          opencl.executor.Executor.benchmark(kernelJNI, creator.getLocal0((int) (1 * MB)), creator.getLocal1((int) (1 * MB)), creator.getLocal2((int) (1 * MB)),
          creator.getGlobal0((int) (1 * MB)), creator.getGlobal1((int) (1 * MB)), creator.getGlobal2((int) (1 * MB)), creator.createArgs((int) (1 * MB)), numberExecutions, 0);
		  //opencl.executor.Executor.benchmark(kernelJNI2, creator.getLocal0((int) (1 * MB)), creator.getLocal1((int) (1 * MB)), creator.getLocal2((int) (1 * MB)),
          //creator.getGlobal0((int) (1 * MB)), creator.getGlobal1((int) (1 * MB)), creator.getGlobal2((int) (1 * MB)), creator.createArgs((int) (1 * MB)), numberExecutions, 0);

        // Start run for every dataSize
        for (int i = 0; i < dataSizesBytes.length; i++) {
            long dataSize = dataSizesBytes[i];

            if (dataSize <= 0)
                throw new IllegalArgumentException();

            int dataLength = creator.getDataLength(dataSize);
			
			System.out.println("Benchmarking with " + humanReadableByteCountBin(dataSize) + "...");

			int local0 = creator.getLocal0(dataLength);
			int local1 = creator.getLocal1(dataLength);
			int local2 = creator.getLocal2(dataLength);
			int global0 = creator.getGlobal0(dataLength);
			int global1 = creator.getGlobal1(dataLength);
			int global2 = creator.getGlobal2(dataLength);
			KernelArg[] args = creator.createArgs(dataLength);

			totalStart = System.currentTimeMillis();

            double[] resultI = opencl.executor.Executor.benchmark(kernelJNI, local0, local1, local2, global0, global1, global2, 
																  args, numberExecutions, 0);

			total[i] = (System.currentTimeMillis() - totalStart) / numberExecutions;
            /*
			// Prepare grid reduces
			dataLength = creator.getGlobal0(dataLength);
			System.out.println("Datalength = " + dataLength);
			
			while (dataLength > 1) {
				local0 = creator.getLocal0(dataLength);
				global0 = creator.getGlobal0(dataLength);
				args = creator.createArgs(dataLength);
				
				totalStart = System.currentTimeMillis();
				
				// Simulate grid reduce
				double[] resultI2 = opencl.executor.Executor.benchmark(kernelJNI2, local0, 1, 1, global0, 1, 1, args, numberExecutions, 0);
			
				total2 = (System.currentTimeMillis() - totalStart) / numberExecutions;
				
				dataLength = creator.getGlobal0(dataLength);
				
				// Add average times
				total[i] += total2;
				for (int k = 0; k < numberExecutions; k++) {
					resultI[i] += resultI2[i];
				}
			}*/

            for (int j = 0; j < numberExecutions; j++)
                result[i] += resultI[j];

            result[i] /= (double) numberExecutions;
        }

        // Absolute time Measurement
        long dt = System.currentTimeMillis() - t0;

        BenchmarkResult r = new BenchmarkResult(numberExecutions, dataSizesBytes, result, total, kernelName, dt);

        System.out.println(r);
    }

    /**
     * Abstract class for generate KernelArgs with a specific size.
     */
    public static abstract class KernelArgCreator {
        /**
         * Returns the length of the data (number of elements).
         *
         * @param dataSizeBytes size of data in bytes
         * @return length of the data
         */
        public abstract int getDataLength(long dataSizeBytes);

        /**
         * Generate KernelArgs.
         *
         * @param dataLength length of the data (number of elements)
         * @return KernelArgs
         */
        public abstract KernelArg[] createArgs(int dataLength);

        /**
         * Returns the number of grids for kernellaunch in first dimension.
         *
         * @param dataLength length of the data (number of elements)
         * @return number of grids for kernellaunch in first dimension
         */
        public abstract int getLocal0(int dataLength);

        /**
         * Returns the number of grids for kernellaunch in second dimension.
         *
         * @param dataLength length of the data (number of elements)
         * @return number of grids for kernellaunch in second dimension
         */
        public int getLocal1(int dataLength) {
            return 1;
        }

        /**
         * Returns the number of grids for kernellaunch in third dimension.
         *
         * @param dataLength length of the data (number of elements)
         * @return number of grids for kernellaunch in third dimension
         */
        public int getLocal2(int dataLength) {
            return 1;
        }

        /**
         * Returns the number of blocks for kernellaunch in first dimension.
         *
         * @param dataLength length of the data (number of elements)
         * @return number of blocks for kernellaunch in first dimension
         */
        public abstract int getGlobal0(int dataLength);

        /**
         * Returns the number of blocks for kernellaunch in second dimension.
         *
         * @param dataLength length of the data (number of elements)
         * @return number of blocks for kernellaunch in second dimension
         */
        public int getGlobal1(int dataLength) {
            return 1;
        }

        /**
         * Returns the number of blocks for kernellaunch in third dimension.
         *
         * @param dataLength length of the data (number of elements)
         * @return number of blocks for kernellaunch in third dimension
         */
        public int getGlobal2(int dataLength) {
            return 1;
        }
    }

    /**
     * Class representing the result of a benchmark-test.
     */
    public static class BenchmarkResult {
        private final int numberExecutions;
        private final long[] dataSizes;
        private final double[] average;
		private final double[] totalAverage;
        private final String kernelName;
        private final long duration;


        protected BenchmarkResult(int numberExecutions, long[] dataSizes, double[] average, double[] totalAverage,
                                  String kernelName, long duration) {
            this.numberExecutions = numberExecutions;
            this.dataSizes = dataSizes;
            this.average = average;
			this.totalAverage = totalAverage;
            this.kernelName = kernelName;
            this.duration = duration;
        }

        /**
         * Returns the number of executions for the kernel for every data size.
         *
         * @return number of executions
         */
        public int getNumberExecutions() {
            return numberExecutions;
        }

        /**
         * Returns the data sizes of the kernel arguments, which was tested, in bytes.
         *
         * @return data sizes, which was tested
         */
        public long[] getDataSizes() {
            return dataSizes;
        }

        /**
         * Returns the average KernelTimes for one kernel execution for every datasize.
         *
         * @return average KernelTimes for one kernel execution for every datasize
         */
        public double[] getAverage() {
            return average;
        }
		
		/**
         * Returns the average total KernelTimes for one kernel execution for every datasize.
         *
         * @return average KernelTimes for one kernel execution for every datasize
         */
        public double[] getTotalAverage() {
            return totalAverage;
        }

        /**
         * Returns the name of the tested kernel.
         *
         * @return name of the tested kernel
         */
        public String getKernelName() {
            return kernelName;
        }

        @Override
        public String toString() {
            StringBuffer buffer = new StringBuffer(200);
            buffer.append("\nBenchmark " + kernelName + "-Kernel");

            buffer.append("  Datasize  Result (Average)\n");

            // For every dataSize: Append average for one kernel execution
            for (int i = 0; i < dataSizes.length; i++) {
                String dataSize = "" + humanReadableByteCountBin(dataSizes[i]);
                while (dataSize.length() < 10)
                    dataSize = " " + dataSize;

                String result = "execution-time: " + average[i] + " ms";
				String result2 = "total-time: " + totalAverage[i] + " ms";

                buffer.append(dataSize);
                buffer.append(": ");
                buffer.append(result);
                buffer.append("\n");
				buffer.append(result2);
                buffer.append("\n");
            }

            buffer.append("\n");
            buffer.append("Benchmark exeuted in " + (duration/1000f) + " s");

            return buffer.toString();
        }

        public static String humanReadableByteCountBin(long bytes) {
            return bytes < 1024L ? bytes + " B"
                    : bytes <= 0xfffccccccccccccL >> 40 ? String.format("%.1f KiB", bytes / 0x1p10)
                    : bytes <= 0xfffccccccccccccL >> 30 ? String.format("%.1f MiB", bytes / 0x1p20)
                    : String.format("%.1f GiB", bytes / 0x1p30);
        }
    }

    /**
     * Reads a file and returns the content of the file as string.
     *
     * @param filename name of a file in current path or complete filename including
     *                 path to file
     * @return string with the filecontent
     * @throws IOException
     */
    public static String loadFile(String filename) throws IOException {
        assert (filename != null);

        return new String(Files.readAllBytes(new File(filename).toPath()));
    }
}

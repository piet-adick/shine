
import opencl.executor.Kernel;
import opencl.executor.KernelArg;
import opencl.executor.KernelTime;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.text.DecimalFormat;

public class OpenCLBenchmarkUtilsTotal {

    public static BenchmarkResult benchmark(String kernelName, String buildOptions, int numberExecutions, KernelArgCreator creator, long... dataSizesBytes) throws IOException {
        if (dataSizesBytes == null)
            throw new NullPointerException();
        if (dataSizesBytes.length == 0)
            throw new IllegalArgumentException("not data sizes specificated");
        if (numberExecutions <= 0)
            throw new IllegalArgumentException("illegal number of executions: " + numberExecutions);

        String kernelString = loadFile(kernelName);

        // Absolute time Measurement
        long t0 = System.currentTimeMillis();

        // Create and compile Kernel
        Kernel kernelJNI = opencl.executor.Kernel.create(loadFile("kernels/" + kernelName+".cu"), kernelName, buildOptions);

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

            // Destroy corresponding C-Objects
            for (KernelArg arg : args) {
                arg.dispose();
            }
        }

        // Absolute time Measurement
        long dt = System.currentTimeMillis() - t0;

        return new BenchmarkResult(numberExecutions, dataSizesBytes, result, kernelName, dt);
    }

    /**
     * Class representing the result of a benchmark-test.
     */
    public static class BenchmarkResult {
        /**
         * Benchmark a CUDA kernel.
         *
         * @param kernelString      string containing the CUDA kernelcode
         * @param kernelName        name of the kernel
         * @param options           options for the nvtrc compiler
         * @param device            device on which the benchmark-test should be
         *                          executed
         * @param templateParameter array of templateParameters or an empty array if the
         *                          kernel do not contains template parameters
         * @param numberExecutions  number of executions for the kernel
         * @param creator           KernelArgCreator for creating KernelArgs for the
         *                          kernel
         * @param dataSizesBytes    data sizes of the kernel arguments, which should be
         *                          tested, in bytes
         * @return result of benchmark-test
         */
        private final String deviceInformation;
        private final int numberExecutions;
        private final long[] dataSizes;
        private final KernelTime[][] result;
        private final KernelTime[] average;
        private final String kernelName;
        private final long testDuration;

        protected BenchmarkResult(int numberExecutions, long[] dataSizes, KernelTime[][] result,
                                  String kernelName, long testDuration) {
            this.numberExecutions = numberExecutions;
            this.dataSizes = dataSizes;
            this.result = result;
            this.kernelName = kernelName;
            this.testDuration = testDuration;

            deviceInformation = "";

            // Compute Average
            average = new KernelTime[result.length];
            for (int i = 0; i < dataSizes.length; i++) {
                double upload = 0;
                double download = 0;
                double launch = 0;
                double total = 0;

                for (int j = 0; j < numberExecutions; j++) {
                    upload += result[i][j].getUpload();
                    download += result[i][j].getDownload();
                    launch += result[i][j].getLaunch();
                    total += result[i][j].getTotal();
                }

                average[i] = new KernelTime((float) (upload / numberExecutions), (float) (download / numberExecutions),
                        (float) (launch / numberExecutions), (float) (total / numberExecutions));
            }
        }

        /**
         * Create a new result of benchmark-test.
         *
         * @param deviceInformation String with deviceInformation
         * @param numberExecutions  number of executions for the kernel for every data
         *                          size
         * @param dataSizes         data sizes of the kernel arguments, which was
         *                          tested, in bytes
         * @param result            KernelTimes for every kernel execution for every
         *                          datasize
         * @param average           average of result
         * @param kernelName        name of the tested kernel
         * @param testDuration      duration of the test in milliseconds
         */
        private BenchmarkResult(String deviceInformation, int numberExecutions, long[] dataSizes, KernelTime[][] result,
                                KernelTime[] average, String kernelName, long testDuration) {
            this.deviceInformation = deviceInformation;
            this.numberExecutions = numberExecutions;
            this.dataSizes = dataSizes;
            this.result = result;
            this.average = average;
            this.kernelName = kernelName;
            this.testDuration = testDuration;
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
         * Returns the KernelTimes for every kernel execution for every datasize.
         *
         * @return KernelTimes for kernel executions
         */
        public KernelTime[][] getResult() {
            return result;
        }

        /**
         * Returns the average KernelTimes for one kernel execution for every datasize.
         *
         * @return average KernelTimes for one kernel execution for every datasize
         */
        public KernelTime[] getAverage() {
            return average;
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
            buffer.append(deviceInformation + "\n");

            buffer.append("  Datasize  Result (Average)\n");

            // For every dataSize: Append average for one kernel execution
            for (int i = 0; i < dataSizes.length; i++) {
                String dataSize = "" + humanReadableByteCountBin(dataSizes[i]);
                while (dataSize.length() < 10)
                    dataSize = " " + dataSize;

                String result = average[i].toString();

                buffer.append(dataSize);
                buffer.append(": ");
                buffer.append(result);
                buffer.append("\n");
            }

            // Absolute execution-time of the test
            DecimalFormat df = new DecimalFormat();
            String time = KernelTime.humanReadableMilliseconds(df, testDuration);
            df.setMaximumFractionDigits(1);
            String[] s = time.split(" ");

            if (s.length == 3)
                buffer.append("\nBenchmark-Duration: " + df.format(Double.parseDouble(s[0])) + " " + s[2] + "\n");
            else
                buffer.append("\nBenchmark-Duration: " + df.format(Double.parseDouble(s[0])) + " " + s[1] + "\n");

            return buffer.toString();
        }

        /**
         * Adds the result of this Benchmark to another BenchmarkResult.
         *
         * @param benchmark BenchmarkResult, which should be added
         * @return sum of the benchmarks
         */
        public BenchmarkResult addBenchmarkResult(BenchmarkResult benchmark) {
            if (numberExecutions != benchmark.numberExecutions)
                throw new IllegalArgumentException("Both benchmark result must have the same number of executions");
            for (int i = 0; i < dataSizes.length; i++)
                if (dataSizes[i] != benchmark.dataSizes[i])
                    throw new IllegalArgumentException("Both benchmark result must have the same dataSizes");

            String deviceInformation;
            if (this.deviceInformation.equals(benchmark.deviceInformation))
                deviceInformation = this.deviceInformation;
            else
                deviceInformation = this.deviceInformation + " and " + benchmark.deviceInformation;

            String kernelName = this.kernelName + " and " + benchmark.kernelName;
            long testDuration = this.testDuration + benchmark.testDuration;

            KernelTime[][] result = new KernelTime[dataSizes.length][numberExecutions];
            KernelTime[] average = new KernelTime[dataSizes.length];

            for (int i = 0; i < dataSizes.length; i++) {
                for (int j = 0; j < numberExecutions; j++) {
                    result[i][j] = this.result[i][j].addKernelTime(benchmark.result[i][j]);
                }

                average[i] = this.average[i].addKernelTime(benchmark.average[i]);
            }

            return new BenchmarkResult(deviceInformation, this.numberExecutions, this.dataSizes, result, average,
                    kernelName, testDuration);
        }

        static String humanReadableByteCountBin(long bytes) {
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

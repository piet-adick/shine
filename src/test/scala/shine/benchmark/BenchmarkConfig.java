public class BenchmarkConfig {
    final static boolean DEBUG = true;

    final static long KB = 1024;
    final static long MB = 1024 * 1024;


    public static int numberExecutionsWarmUp;
    public static int numberExecutions;

    public static long[] dataSizesGEMM;
    public static long[] dataSizesReduce;

    static {
        //Use small numbers for fast debug
        if (DEBUG){
            numberExecutionsWarmUp = 1;
            numberExecutions = 1;
            dataSizesGEMM = new long[]{1 * KB, 4 * MB};
            dataSizesReduce = new long[]{1 * KB, 4 * MB};
        }
        //For final benchmark
        else {
            numberExecutionsWarmUp = 30;
            numberExecutions = 50;
            dataSizesGEMM = new long[] { 1 * KB, 4 * KB, 16 * KB, 64 * KB, 256 * KB, 1 * MB, 4 * MB,
                    16 * MB, 64 * MB, 256 * MB, 1024 * MB };
            dataSizesReduce = new long[] { 1 * KB, 4 * KB, 16 * KB, 64 * KB, 256 * KB, 1 * MB, 4 * MB,
                    16 * MB, 64 * MB, 256 * MB, 1024 * MB };
        }
    }
}

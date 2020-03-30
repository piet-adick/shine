public class BenchmarkConfig {
    final static boolean DEBUG = true;

    final static long KB = 1024;
    final static long MB = 1024 * 1024;


    public static int numberExecutionsWarmUp;
    public static long warmUpSize;

    public static int numberExecutions;
    public static long[] dataSizesGEMMYacx;
    public static long[] dataSizesGEMMCLBlastKepler;
    public static long[] dataSizesKeplerBest;
    public static long[] dataSizesKeplerS;
    public static long[] dataSizesReduceYacx;
    public static long[] dataSizesReduceOpenCL;

    static {
        //Use small numbers for fast debug
        if (DEBUG){
            numberExecutionsWarmUp = 1;
            numberExecutions = 1;
            dataSizesGEMMYacx = new long[] { 1 * KB, 4 * MB};
            dataSizesGEMMCLBlastKepler = dimToDataLength(10, 10);
            dataSizesKeplerBest = dimToDataLength(10, 10);
            dataSizesKeplerS = dimToDataLength(10, 10);
            dataSizesReduceYacx = new long[] { 1 * KB, 4 * MB};
            dataSizesReduceOpenCL = new long[] { 1 * KB, 4 * MB};
            warmUpSize = dimToDataLength(512)[0];
        }
        //For final benchmark
        else {
            numberExecutionsWarmUp = 30;
            numberExecutions = 50;
            dataSizesGEMMYacx = new long[] { 1 * KB, 4 * KB, 16 * KB, 64 * KB, 256 * KB, 1 * MB, 4 * MB,
                    16 * MB, 64 * MB, 256 * MB, 1024 * MB };
            dataSizesReduceYacx = new long[] { 1 * KB, 4 * KB, 16 * KB, 64 * KB, 256 * KB, 1 * MB, 4 * MB,
                    16 * MB, 64 * MB, 256 * MB, 1024 * MB };
            warmUpSize = 256 * MB;
        }
    }

    static long[] dimToDataLength(int... dims){
        long[] dataSizes = new long[dims.length];
        for (int i = 0; i < dims.length; i++){
            dataSizes[i] = dims[i]*dims[i]*4;
        }

        return dataSizes;
    }
}

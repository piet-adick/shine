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
            dataSizesGEMMYacx = generateDataSizes(4, 2);
            //Dimension: Vielfache von 128
            dataSizesGEMMCLBlastKepler = dataSizesGEMMYacx;
            //Dimension: Vielfache von 128
            dataSizesKeplerBest = dataSizesGEMMYacx;
            dataSizesKeplerS = dataSizesGEMMYacx;
            dataSizesReduceYacx = dataSizesGEMMYacx;
            //Dimension: Vielfache von 128
            dataSizesReduceOpenCL = dataSizesGEMMYacx;
            warmUpSize = dimToDataLength(256)[0];
        }
        //For final benchmark
        else {
            numberExecutionsWarmUp = 30;
            numberExecutions = 50;
            dataSizesGEMMYacx = generateDataSizes(2, 70);
            //Dimension: Vielfache von 128
            dataSizesGEMMCLBlastKepler = dataSizesGEMMYacx;
            //Dimension: Vielfache von 128
            dataSizesKeplerBest = dataSizesGEMMYacx;
            dataSizesKeplerS = dataSizesGEMMYacx;
            dataSizesReduceYacx = dataSizesGEMMYacx;
            //Dimension: Vielfache von 128
            dataSizesReduceOpenCL = dataSizesGEMMYacx;
            warmUpSize = dimToDataLength(1024*32)[0];
        }
    }

    static long[] generateDataSizes(int incrementFactor, int numberDataSizes){
        int[] dims = new int[numberDataSizes];

        for (int i = 0; i < dims.length; i++){
            dims[i] = 128 + i*128*incrementFactor;
        }

        return dimToDataLength(dims);
    }

    static long[] dimToDataLength(int... dims){
        long[] dataSizes = new long[dims.length];
        for (int i = 0; i < dims.length; i++){
            dataSizes[i] = dims[i]*dims[i]*4;
        }

        return dataSizes;
    }
}

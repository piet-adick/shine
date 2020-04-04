public class BenchmarkConfig {
    final static boolean DEBUG = false;

    public static int numberExecutionsWarmUp;
    public static long warmUpSize;

    public static int numberExecutions;
    public static long[] dataSizesGEMMYacx;
    public static long[] dataSizesGEMMCLBlastKepler;
    public static long[] dataSizesKeplerBest;
    public static long[] dataSizesKeplerS;
    public static long[] dataSizesReduceYacx;
    public static long[] dataSizesReduceOpenCL;
    public static long[] dataSizesVetorAdd;

    static {
        numberExecutionsWarmUp = 40;
        numberExecutions = 80;
        //Padding will be tested automatically
        dataSizesGEMMYacx = generateDataSizes128(2, 70);
        //Dimension must be multiple of 128
        dataSizesGEMMCLBlastKepler = dataSizesGEMMYacx;
        //Dimension must be multiple of 128
        //This is very slowly
        dataSizesKeplerBest = generateDataSizes128(24, 6);
        dataSizesKeplerS = dataSizesGEMMCLBlastKepler;
        dataSizesReduceYacx = generateDataSizes2048(18, 60);
        //Datalength must be multiple of 2048
        //At Datasize of 918MB there was a out-of-memory :(
        dataSizesReduceOpenCL = generateDataSizes2048(18, 33);
        dataSizesVetorAdd = dataSizesGEMMCLBlastKepler;
        warmUpSize = dimToDataLength(1024*8)[0];
    }

    public static void main(String[] args) {
        System.out.println("GEMMBenchmark: " + java.util.Arrays.toString(asMB(dataSizesGEMMYacx)));
        System.out.println("CLBlast: " + java.util.Arrays.toString(asMB(dataSizesGEMMCLBlastKepler)));
        System.out.println("KeplerBest: " + java.util.Arrays.toString(asMB(dataSizesKeplerBest)));
        System.out.println("KeplerS: " + java.util.Arrays.toString(asMB(dataSizesKeplerS)));
        System.out.println("ReduceBenchmark: " + java.util.Arrays.toString(asMB(dataSizesReduceYacx)));
        System.out.println("ReduceOpenCL: " + java.util.Arrays.toString(asMB(dataSizesReduceOpenCL)));
        System.out.println("VetorAdd: " + java.util.Arrays.toString(asMB(dataSizesVetorAdd)));
    }

    static float[] asMB(long[] dataSizes){
        float[] dataSizeMb = new float[dataSizes.length];

        for (int i = 0; i < dataSizeMb.length; i++){
            dataSizeMb[i] = dataSizes[i]/1024f/1024f;
        }

        return dataSizeMb;
    }

    static long[] generateDataSizesPaddingTest(int incrementFactor, int numberDataSizes){
        int[] dims = new int[numberDataSizes*2];

        for (int i = 0; i < dims.length; i += 2){
            dims[i] = 128 + i/2*128*incrementFactor - 1;
            dims[i+1] = 128 + i/2*128*incrementFactor;
        }

        return dimToDataLength(dims);
    }

    static long[] generateDataSizes2048(int incrementFactor, int numberDataSizes){
        long[] dataSizes = new long[numberDataSizes];

        for (int i = 0; i < dataSizes.length; i++){
            dataSizes[i] = 2048l + incrementFactor*2048l*128*i *4l;
        }

        return dataSizes;
    }

    static long[] generateDataSizes128(int incrementFactor, int numberDataSizes){
        int[] dims = new int[numberDataSizes];

        for (int i = 0; i < dims.length; i++){
            dims[i] = 128 + i*128*incrementFactor;
        }

        return dimToDataLength(dims);
    }

    static long[] dimToDataLength(int... dims){
        long[] dataSizes = new long[dims.length];
        for (int i = 0; i < dims.length; i++){
            dataSizes[i] = dims[i]*dims[i]*4l;
        }

        return dataSizes;
    }
}

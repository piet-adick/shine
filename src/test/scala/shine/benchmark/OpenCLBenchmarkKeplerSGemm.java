
import opencl.executor.GlobalArg;
import opencl.executor.KernelArg;
import opencl.executor.ValueArg;

import java.io.IOException;

public class OpenCLBenchmarkKeplerSGemm {

    //Kernelname (kernelcode in file kernelname.cl)
    static String kernel = "keplerSgemm";
    static String options = "";
    static long[] dataSizes = BenchmarkConfig.dataSizesKeplerS;

    //KernelArgeCreator
    static KernelArgCreator creator = new KernelArgCreator(){
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
            int m = (x >= 512) ? x : 512;
            int n = (x >= 256) ? x : 256;
            int k = (x >= 64) ? x : 64;
            float alpha = 1f;
            float beta = 1f;
            float[] aMatrix = new float[m * k];
            float[] bMatrix = new float[k * n];
            float[] cMatrix = new float[m * n];
            for (int i = 0; i < aMatrix.length; i++) {
                aMatrix[i] = (i < x) ? i : 0;
            }
            for (int i = 0; i < bMatrix.length; i++) {
                bMatrix[i] = (i < x) ? i : 0;
            }
            for (int i = 0; i < cMatrix.length; i++) {
                cMatrix[i] = (i < x) ? i : 0;
            }

            GlobalArg aArg = GlobalArg.createInput(aMatrix);
            GlobalArg bArg = GlobalArg.createInput(bMatrix);
            GlobalArg cArg = GlobalArg.createInput(cMatrix);
            GlobalArg outputArg = GlobalArg.createOutput(m * n * 4);
            ValueArg alphaArg = ValueArg.create(alpha);
            ValueArg betaArg = ValueArg.create(beta);
            ValueArg mArg = ValueArg.create(m);
            ValueArg nArg = ValueArg.create(n);
            ValueArg kArg = ValueArg.create(k);

            return new KernelArg[]{aArg, bArg, cArg, alphaArg, betaArg, outputArg, kArg, mArg, nArg};
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
}

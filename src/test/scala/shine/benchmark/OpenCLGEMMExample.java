
import opencl.executor.GlobalArg;
import opencl.executor.Kernel;
import opencl.executor.KernelArg;
import opencl.executor.ValueArg;

import java.io.IOException;

public class OpenCLGEMMExample {
    public static void main(String[] args) throws IOException {
        opencl.executor.Executor.loadAndInit();

        int m = 512;
        int n = 256;
        int k = 64;
        float alpha = 1f;
        float beta = 1f;
        float[] aMatrix = new float[m * k];
        float[] bMatrix = new float[k * n];
        float[] cMatrix = new float[m * n];
        aMatrix[0] = 1;
        aMatrix[1] = 3;
        aMatrix[k] = 2;
        aMatrix[k+1] = 4;
        bMatrix[0] = 1;
        bMatrix[1] = 2;
        bMatrix[n] = 3;
        bMatrix[n+1] = 4;

        // Create Arguments
        GlobalArg aMatrixArg = GlobalArg.createInput(aMatrix);
        // Kernel expects a transposed B matrix so this has to be done here
        GlobalArg bMatrixArg = GlobalArg.createInput(bMatrix);
        GlobalArg cMatrixArg = GlobalArg.createInput(cMatrix);
        GlobalArg dMatrixArg = GlobalArg.createOutput(m * n * 4);
        KernelArg mArg = ValueArg.create(m);
        KernelArg nArg = ValueArg.create(n);
        KernelArg kArg = ValueArg.create(k);
        KernelArg alphaArg = ValueArg.create(alpha);
        KernelArg betaArg = ValueArg.create(beta);
        KernelArg[] kernelArgs = new KernelArg[]{aMatrixArg, bMatrixArg, cMatrixArg, alphaArg, betaArg, dMatrixArg, kArg, mArg, nArg};

        System.out.println();
        System.out.println("aMatrix:");
        printlnMatrix(aMatrix, k);
        System.out.println("bMatrix:");
        printlnMatrix(bMatrix, n);
        System.out.println("cMatrix:");
        printlnMatrix(cMatrix, n);

        Kernel kernel = Kernel.create(OpenCLBenchmarkUtils.loadFile("kernels/keplerBestSgemm.cl"), "keplerBestSgemm", "");

        opencl.executor.Executor.execute(kernel, 32, 8, 1, 512/4, 512/8, 1, kernelArgs);

        // Print Result
        System.out.println("Kernel keplerBestSgemm launched ");
        System.out.println("resultmatrix:");
        printlnMatrix(dMatrixArg.asFloatArray(), n);

        Kernel kernel1 = Kernel.create(OpenCLBenchmarkUtils.loadFile("kernels/keplerSgemm.cl"), "keplerSgemm", "");
        opencl.executor.Executor.execute(kernel1, 32, 8, 1, 256, 128, 1, kernelArgs);

        // Print Result
        System.out.println("Kernel keplerSgemm launched ");
        System.out.println("resultmatrix:");
        printlnMatrix(dMatrixArg.asFloatArray(), n);

        Kernel kernel2 = Kernel.create(OpenCLBenchmarkUtils.loadFile("kernels/clblast_kepler_sgemm.cl"), "clblast_kepler_sgemm", "");
        opencl.executor.Executor.execute(kernel2, 128/8, 8, 1, 512/8, 512/8, 1, kernelArgs);

        // Print Result
        System.out.println("Kernel clblast_kepler_sgemm launched ");
        System.out.println("resultmatrix:");
        printlnMatrix(dMatrixArg.asFloatArray(), n);
    }

    public static void printlnMatrix(float[] matrix, int columns) {
        assert (matrix.length % columns == 0);

        int rows = matrix.length / columns;

        int stringLengthElement = 3;
        for (int i = 0; i < matrix.length; i++)
            if (("" + matrix[i]).length() > stringLengthElement)
                stringLengthElement = ("" + matrix[i]).length();

        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < columns; column++) {
                String elementString = "" + matrix[row * columns + column];

                for (int i = elementString.length(); i < stringLengthElement + 1; i++)
                    System.out.print(" ");

                System.out.print(elementString);
            }

            System.out.println();
        }
        System.out.println();
    }
}

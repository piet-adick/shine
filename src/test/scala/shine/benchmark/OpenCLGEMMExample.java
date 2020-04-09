
import opencl.executor.GlobalArg;
import opencl.executor.Kernel;
import opencl.executor.KernelArg;
import opencl.executor.ValueArg;

import java.io.IOException;

public class OpenCLGEMMExample {
    public static void main(String[] args) throws IOException {
        opencl.executor.Executor.loadAndInit();

        int x = 512;
        int y = 512;
        int z = 512;
        float alpha = 1f;
        float beta = 1f;
        float[] aMatrix = new float[x * y];
        float[] bMatrix = new float[y * z];
        float[] cMatrix = new float[x * z];
        aMatrix[0] = 1;
        aMatrix[1] = 2;
        aMatrix[512] = 3;
        aMatrix[513] = 4;
        bMatrix[0] = 1;
        bMatrix[1] = 2;
        bMatrix[512] = 3;
        bMatrix[513] = 4;

        // Create Arguments
        GlobalArg aMatrixArg = GlobalArg.createInput(aMatrix);
        // Kernel expects a transposed B matrix so this has to be done here
        GlobalArg bMatrixArg = GlobalArg.createInput(bMatrix);
        GlobalArg cMatrixArg = GlobalArg.createInput(cMatrix);
        GlobalArg dMatrixArg = GlobalArg.createOutput(x * z * 4);
        KernelArg mArg = ValueArg.create(x);
        KernelArg nArg = ValueArg.create(y);
        KernelArg kArg = ValueArg.create(z);
        KernelArg alphaArg = ValueArg.create(alpha);
        KernelArg betaArg = ValueArg.create(beta);
        KernelArg[] kernelArgs = new KernelArg[]{aMatrixArg, bMatrixArg, cMatrixArg, alphaArg, betaArg, dMatrixArg, kArg, mArg, nArg};

        System.out.println();
        System.out.println("aMatrix:");
        printlnMatrix(aMatrix, y);
        System.out.println("bMatrix:");
        printlnMatrix(bMatrix, z);
        System.out.println("cMatrix:");
        printlnMatrix(cMatrix, z);

        Kernel kernel = Kernel.create(OpenCLBenchmarkUtils.loadFile("kernels/keplerBestSgemm.cl"), "keplerBestSgemm", "");

        opencl.executor.Executor.execute(kernel, 32, 8, 1, 512/4, 512/8, 1, kernelArgs);

        // Print Result
        System.out.println("Kernel keplerBestSgemm launched ");
        System.out.println("resultmatrix:");
        printlnMatrix(dMatrixArg.asFloatArray(), z);

        Kernel kernel1 = Kernel.create(OpenCLBenchmarkUtils.loadFile("kernels/keplerSgemm.cl"), "keplerSgemm", "");
        opencl.executor.Executor.execute(kernel1, 32, 8, 1, 256, 128, 1, kernelArgs);

        // Print Result
        System.out.println("Kernel keplerSgemm launched ");
        System.out.println("resultmatrix:");
        printlnMatrix(dMatrixArg.asFloatArray(), z);

        Kernel kernel2 = Kernel.create(OpenCLBenchmarkUtils.loadFile("kernels/clblast_kepler_sgemm.cl"), "clblast_kepler_sgemm", "");
        opencl.executor.Executor.execute(kernel2, 128/8, 8, 1, 512/8, 512/8, 1, kernelArgs);

        // Print Result
        System.out.println("Kernel clblast_kepler_sgemm launched ");
        System.out.println("resultmatrix:");
        printlnMatrix(dMatrixArg.asFloatArray(), z);
    }

    public static void printlnMatrix(float[] matrix, int columns) {
        System.out.println("" + matrix[0] + ", " + matrix[1]);
        System.out.println("" + matrix[512] + ", " + matrix[513]);
    }
}

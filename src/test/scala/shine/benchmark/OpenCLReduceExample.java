
import opencl.executor.GlobalArg;
import opencl.executor.Kernel;
import opencl.executor.KernelArg;
import opencl.executor.ValueArg;


import java.io.IOException;

public class OpenCLReduceExample {
    public static void main(String[] args) throws IOException {
        opencl.executor.Executor.loadAndInit();

        int n = 2048 * 128;
		float[] in = new float[n];
        for (int i = 0; i < in.length; i++) {
            in[i] = i + 1;
        }

        GlobalArg outputArg = GlobalArg.createOutput(n * 4 / 2048);
        GlobalArg inputArg = GlobalArg.createInput(in);
        ValueArg nArg = ValueArg.create(n);
		KernelArg[] kernelArgs = new KernelArg[]{outputArg, inputArg, nArg};

        System.out.println("n = " + n);

        Kernel kernel = Kernel.create(OpenCLBenchmarkUtils.loadFile("kernels/nvidiaDerived1.cl"), "nvidiaDerived1", "");

        opencl.executor.Executor.execute(kernel, 128, 1, 1, n, 1, 1, kernelArgs);

        // Print Result
        System.out.println("Output: ");
		float[] out = outputArg.asFloatArray();
		for (int i = 0; i < out.length; i++) {
			System.out.println(i + ": " + out[i]);
		}
		System.out.println();
		
		float result = 0;
		for (int i = 0; i < out.length; i++) {
			result += out[i];
		}
		System.out.println("Final Reduce = " + result);
    }
}
import opencl.executor.KernelArg;

/**
 * Abstract class for generate KernelArgs with a specific size.
 */
public abstract class KernelArgCreator {
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
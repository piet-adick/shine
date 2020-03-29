public class Print {
    public static String humanReadableByteCountBin(long bytes) {
        return bytes < 1024L ? bytes + " B"
                : bytes <= 0xfffccccccccccccL >> 40 ? String.format("%.1f KiB", bytes / 0x1p10)
                : bytes <= 0xfffccccccccccccL >> 30 ? String.format("%.1f MiB", bytes / 0x1p20)
                : String.format("%.1f GiB", bytes / 0x1p30);
    }
}

import crypto from "crypto";
import path from "path";
import fs from "fs/promises";
import { existsSync } from "fs";

export function resolveToAbsolutePath(filePath: string): string {
    console.log("filepath", filePath)
    let cleanPath = filePath.split("?")[0];
    if (cleanPath.startsWith("/_astro/")) {
        return path.resolve(path.join(process.cwd(), "dist", cleanPath));
    }
    if (cleanPath.startsWith("/@fs")) {
        cleanPath = cleanPath.replace("/@fs", "");
    }
    const isProjectRelative = cleanPath.startsWith("/src/");
    const isAbsolute =
        !isProjectRelative &&
        (cleanPath.startsWith("/Users/") || path.isAbsolute(cleanPath));
    return isAbsolute
        ? cleanPath
        : path.resolve(path.join(process.cwd(), cleanPath));
}

export async function getFileStatsAndHash(
    filePath: string,
): Promise<{ size: number; mtime: number; hash: string }> {
    try {
        // Clean up string mutations mimicking the Astro component loader
        const targetPath = resolveToAbsolutePath(filePath);
        const stat = await fs.stat(targetPath);
        const size = stat.size;
        const mtime = stat.mtimeMs;

        const sampleSize = Math.min(size, 4096);
        const handle = await fs.open(targetPath, "r");
        const buffer = Buffer.alloc(sampleSize);

        await handle.read(buffer, 0, sampleSize, 0);
        await handle.close();

        const sampleHash = crypto.createHash("md5").update(buffer).digest("hex");

        return { size, mtime, hash: sampleHash };
    } catch (err) {
        console.error(
            `[vite-image-pipeline] Failed to generate stats and hash for ${filePath} ${resolveToAbsolutePath(filePath)}:`,
            err,
        );
        throw err;
    }
}

export async function loadCache<T>(filepath: string): Promise<Record<string, T>> {
    if (existsSync(filepath)) {
        try {
            const raw = await fs.readFile(filepath, "utf-8");
            return JSON.parse(raw) as Record<string, T>;
        } catch (e) {
            console.error(
                `[vite-image-pipeline] Failed to load cache at ${filepath}.`,
                e,
            );
        }
    }
    return {};
}

export async function saveCache(filepath: string, cache: Record<string, any> | null) {
    if (!cache) return;
    try {
        await fs.mkdir(path.dirname(filepath), { recursive: true });
        await fs.writeFile(
            filepath,
            JSON.stringify(cache, null, 2),
            "utf-8",
        );
    } catch (err) {
        console.error(
            "[vite-image-pipeline] Failed to sync cache:",
            err,
        );
    }
}
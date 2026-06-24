// image-pipeline.ts
import crypto from "crypto";
import { exiftool, type Tags } from "exiftool-vendored";
import { pipeline, env, RawImage } from "@huggingface/transformers";
import fs from "fs/promises";
import { existsSync } from "fs";
import path from "path";
import sharp from "sharp";

export interface CacheEntry<T> {
  mtime: number;
  size: number;
  hash: string;
  data: T;
}

export interface MetadataCacheSchema {
  [filePath: string]: CacheEntry<Tags>;
}

export interface EmbeddingCacheSchema {
  [filePath: string]: CacheEntry<number[]>;
}

// ─── NEW SCHEMA SCHEMAS ─────────────────────────────────────────────
export interface ColorCacheSchema {
  [filePath: string]: CacheEntry<{ r: number; g: number; b: number }>;
}

export interface BlurCacheSchema {
  [filePath: string]: CacheEntry<string>; // Base64 Data URI string
}

let globalPipelinePromise: Promise<any> | null = null;

export class ImagePipelineEngine {
  private metadataCachePath: string;
  private embeddingCachePath: string;
  private colorCachePath: string;
  private blurCachePath: string;
  private modelCachePath: string;

  private metadataCache: MetadataCacheSchema | null = null;
  private embeddingCache: EmbeddingCacheSchema | null = null;
  private colorCache: ColorCacheSchema | null = null;
  private blurCache: BlurCacheSchema | null = null;

  private modelName: string;
  private visionPipeline: any = null;
  private ML_BATCH_SIZE: number;

  private metadataLock: Promise<any> = Promise.resolve();
  private embeddingLock: Promise<any> = Promise.resolve();
  private colorLock: Promise<any> = Promise.resolve();
  private blurLock: Promise<any> = Promise.resolve();

  constructor(
    options: {
      modelName?: string;
      metadataCachePath?: string;
      embeddingCachePath?: string;
      colorCachePath?: string;
      blurCachePath?: string;
      batchSize?: number;
      modelCachePath?: string;
    } = {},
  ) {
    this.modelName = options.modelName || "Xenova/clip-vit-base-patch32";
    this.metadataCachePath =
      options.metadataCachePath ||
      path.resolve(".image-pipeline/metadata-cache.json");
    this.embeddingCachePath =
      options.embeddingCachePath ||
      path.resolve(".image-pipeline/embedding-cache.json");
    this.colorCachePath =
      options.colorCachePath ||
      path.resolve(".image-pipeline/color-cache.json");
    this.blurCachePath =
      options.blurCachePath || path.resolve(".image-pipeline/blur-cache.json");
    this.ML_BATCH_SIZE = options.batchSize || 4;
    this.modelCachePath =
      options.modelCachePath || path.resolve(".image-pipeline/models");
  }

  private async loadMetadataCache() {
    if (existsSync(this.metadataCachePath)) {
      try {
        const raw = await fs.readFile(this.metadataCachePath, "utf-8");
        this.metadataCache = JSON.parse(raw);
        return;
      } catch (e) {
        console.error(
          `[astro-image-pipeline] Failed to load metadata cache at ${this.metadataCachePath}.`,
          e,
        );
      }
    }
    this.metadataCache = {};
  }

  private async loadEmbeddingCache() {
    if (existsSync(this.embeddingCachePath)) {
      try {
        const raw = await fs.readFile(this.embeddingCachePath, "utf-8");
        this.embeddingCache = JSON.parse(raw);
        return;
      } catch (e) {
        console.error(
          `[astro-image-pipeline] Failed to load embedding cache at ${this.embeddingCachePath}.`,
          e,
        );
      }
    }
    this.embeddingCache = {};
  }

  private async loadColorCache() {
    if (existsSync(this.colorCachePath)) {
      try {
        const raw = await fs.readFile(this.colorCachePath, "utf-8");
        this.colorCache = JSON.parse(raw);
        return;
      } catch (e) {
        console.error(
          `[astro-image-pipeline] Failed to load color cache at ${this.colorCachePath}.`,
          e,
        );
      }
    }
    this.colorCache = {};
  }

  private async loadBlurCache() {
    if (existsSync(this.blurCachePath)) {
      try {
        const raw = await fs.readFile(this.blurCachePath, "utf-8");
        this.blurCache = JSON.parse(raw);
        return;
      } catch (e) {
        console.error(
          `[astro-image-pipeline] Failed to load blur cache at ${this.blurCachePath}.`,
          e,
        );
      }
    }
    this.blurCache = {};
  }

  private resolveToAbsolutePath(filePath: string): string {
    let cleanPath = filePath.split("?")[0];
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

  public async getPipeline() {
    if (this.visionPipeline) return this.visionPipeline;
    if (!globalPipelinePromise) {
      globalPipelinePromise = (async () => {
        env.useFSCache = true;
        env.cacheDir = path.resolve(this.modelCachePath);
        if (env.backends?.onnx?.wasm) {
          env.backends.onnx.wasm.numThreads =
            process.env.NODE_ENV === "production" ? 1 : 4;
        }
        env.allowLocalModels = true;
        try {
          return await pipeline("image-feature-extraction", this.modelName, {
            device: "cpu",
          });
        } catch {
          env.allowLocalModels = false;
          return await pipeline("image-feature-extraction", this.modelName, {
            device: "cpu",
          });
        }
      })();
    }
    this.visionPipeline = await globalPipelinePromise;
    console.log(`[astro-image-pipeline] Image embedding pipeline loaded.`);
    return this.visionPipeline;
  }

  private async getFileStatsAndHash(
    filePath: string,
  ): Promise<{ size: number; mtime: number; hash: string }> {
    try {
      // Clean up string mutations mimicking the Astro component loader
      const targetPath = this.resolveToAbsolutePath(filePath);
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
        `[astro-image-pipeline] Failed to generate stats and hash for ${filePath}:`,
        err,
      );
      throw err;
    }
  }

  private async saveMetadataCache() {
    try {
      await fs.mkdir(path.dirname(this.metadataCachePath), { recursive: true });
      await fs.writeFile(
        this.metadataCachePath,
        JSON.stringify(this.metadataCache, null, 2),
        "utf-8",
      );
    } catch (err) {
      console.error(
        "[astro-image-pipeline] Failed to sync metadata cache:",
        err,
      );
    }
  }

  private async saveEmbeddingCache() {
    try {
      await fs.mkdir(path.dirname(this.embeddingCachePath), {
        recursive: true,
      });
      await fs.writeFile(
        this.embeddingCachePath,
        JSON.stringify(this.embeddingCache, null, 2),
        "utf-8",
      );
    } catch (err) {
      console.error(
        "[astro-image-pipeline] Failed to sync embedding cache:",
        err,
      );
    }
  }

  private async saveColorCache() {
    try {
      await fs.mkdir(path.dirname(this.colorCachePath), { recursive: true });
      await fs.writeFile(
        this.colorCachePath,
        JSON.stringify(this.colorCache, null, 2),
        "utf-8",
      );
    } catch (err) {
      console.error("[astro-image-pipeline] Failed to sync color cache:", err);
    }
  }

  private async saveBlurCache() {
    try {
      await fs.mkdir(path.dirname(this.blurCachePath), { recursive: true });
      await fs.writeFile(
        this.blurCachePath,
        JSON.stringify(this.blurCache, null, 2),
        "utf-8",
      );
    } catch (err) {
      console.error("[astro-image-pipeline] Failed to sync blur cache:", err);
    }
  }

  // ─── GET DOMINANT COLORS METRIC (PARALLELIZED) ────────────────────
  public async getImageColors(
    filePaths: string[],
    options?: { signal?: AbortSignal },
  ): Promise<Record<string, { r: number; g: number; b: number }>> {
    const currentLock = this.colorLock.catch(() => {});
    const executionPromise = (async () => {
      await currentLock;
      if (options?.signal?.aborted)
        throw new DOMException("Aborted", "AbortError");

      const [, ...fileDescriptorsArray] = await Promise.all([
        this.colorCache ? Promise.resolve() : this.loadColorCache(),
        ...filePaths.map(async (filePath) => {
          const stats = await this.getFileStatsAndHash(filePath);
          return { filePath, ...stats };
        }),
      ]);

      if (!this.colorCache)
        throw Error(`[astro-image-pipeline] Failed to load color cache.`);

      const results: Record<string, { r: number; g: number; b: number }> = {};
      const distinctDescriptorsToProcess: typeof fileDescriptorsArray = [];
      let cacheHits = 0;

      for (const desc of fileDescriptorsArray) {
        const cached = this.colorCache[desc.filePath];
        if (
          cached &&
          cached.mtime === desc.mtime &&
          cached.size === desc.size &&
          cached.hash === desc.hash
        ) {
          results[desc.filePath] = cached.data;
          cacheHits++;
        } else {
          distinctDescriptorsToProcess.push(desc);
        }
      }

      if (distinctDescriptorsToProcess.length === 0) return results;
      if (options?.signal?.aborted)
        throw new DOMException("Aborted", "AbortError");

      await Promise.all(
        distinctDescriptorsToProcess.map(async (desc) => {
          try {
            const sharpInstance = sharp(
              this.resolveToAbsolutePath(desc.filePath),
            );
            const { dominant } = await sharpInstance.stats();
            const payload = { r: dominant.r, g: dominant.g, b: dominant.b };

            this.colorCache![desc.filePath] = {
              mtime: desc.mtime,
              size: desc.size,
              hash: desc.hash,
              data: payload,
            };
            results[desc.filePath] = payload;
          } catch (err) {
            console.error(
              `[astro-image-pipeline] Color Generation Failure for ${desc.filePath}:`,
              err,
            );
          }
        }),
      );

      await this.saveColorCache();
      return results;
    })();

    this.colorLock = executionPromise;
    return await executionPromise;
  }

  // ─── GET BLUR PLACEHOLDERS METRIC (PARALLELIZED) ──────────────────
  public async getImageBlurPlaceholders(
    filePaths: string[],
    options?: { signal?: AbortSignal },
  ): Promise<Record<string, string>> {
    const currentLock = this.blurLock.catch(() => {});
    const executionPromise = (async () => {
      await currentLock;
      if (options?.signal?.aborted)
        throw new DOMException("Aborted", "AbortError");

      const [, ...fileDescriptorsArray] = await Promise.all([
        this.blurCache ? Promise.resolve() : this.loadBlurCache(),
        ...filePaths.map(async (filePath) => {
          const stats = await this.getFileStatsAndHash(filePath);
          return { filePath, ...stats };
        }),
      ]);

      if (!this.blurCache)
        throw Error(`[astro-image-pipeline] Failed to load blur cache.`);

      const results: Record<string, string> = {};
      const distinctDescriptorsToProcess: typeof fileDescriptorsArray = [];
      let cacheHits = 0;

      for (const desc of fileDescriptorsArray) {
        const cached = this.blurCache[desc.filePath];
        if (
          cached &&
          cached.mtime === desc.mtime &&
          cached.size === desc.size &&
          cached.hash === desc.hash
        ) {
          results[desc.filePath] = cached.data;
          cacheHits++;
        } else {
          distinctDescriptorsToProcess.push(desc);
        }
      }

      if (distinctDescriptorsToProcess.length === 0) return results;
      if (options?.signal?.aborted)
        throw new DOMException("Aborted", "AbortError");

      await Promise.all(
        distinctDescriptorsToProcess.map(async (desc) => {
          try {
            const sharpInstance = sharp(
              this.resolveToAbsolutePath(desc.filePath),
            );

            const buffer = await sharpInstance.resize(20).blur(1).toBuffer();
            const base64 = buffer.toString("base64");

            // Infer simple fallback MIME type extensions cleanly
            const ext = path
              .extname(desc.filePath)
              .toLowerCase()
              .replace(".", "");
            const mimeType =
              ext === "png"
                ? "image/png"
                : ext === "webp"
                  ? "image/webp"
                  : "image/jpeg";
            const dataUri = `data:${mimeType};base64,${base64}`;

            this.blurCache![desc.filePath] = {
              mtime: desc.mtime,
              size: desc.size,
              hash: desc.hash,
              data: dataUri,
            };
            results[desc.filePath] = dataUri;
          } catch (err) {
            console.error(
              `[astro-image-pipeline] Blur Generation Failure for ${desc.filePath}:`,
              err,
            );
          }
        }),
      );

      await this.saveBlurCache();
      return results;
    })();

    this.blurLock = executionPromise;
    return await executionPromise;
  }

  // ─── UNTOUCHED PRE-EXISTING PIPELINE LOGIC ────────────────────────
  public async getMetadata(
    filePaths: string[],
    options?: { signal?: AbortSignal },
  ): Promise<Record<string, Tags>> {
    const currentLock = this.metadataLock.catch(() => {});
    const executionPromise = (async () => {
      await currentLock;
      if (options?.signal?.aborted)
        throw new DOMException("Aborted", "AbortError");

      if (!this.metadataCache) await this.loadMetadataCache();
      const [, ...fileDescriptorsArray] = await Promise.all([
        ...filePaths.map(async (filePath) => {
          const stats = await this.getFileStatsAndHash(filePath);
          return { filePath, ...stats };
        }),
      ]);

      if (!this.metadataCache)
        throw Error(`[astro-image-pipeline] Failed to load metadata cache.`);

      const results: Record<string, Tags> = {};
      const distinctFilePaths = new Set();
      const distinctDescriptorsToProcess: typeof fileDescriptorsArray = [];
      let cacheHits = 0;

      for (const desc of fileDescriptorsArray) {
        const cached = this.metadataCache[desc.filePath];
        if (
          cached &&
          cached.mtime === desc.mtime &&
          cached.size === desc.size &&
          cached.hash === desc.hash
        ) {
          results[desc.filePath] = cached.data;
          cacheHits++;
        } else {
          if (!distinctFilePaths.has(desc.filePath)) {
            distinctDescriptorsToProcess.push(desc);
            distinctFilePaths.add(desc.filePath);
          }
        }
      }

      console.log(
        "[astro-image-pipeline] Metadata cache hits:",
        cacheHits,
        "/",
        filePaths.length,
      );

      if (distinctDescriptorsToProcess.length === 0) return results;
      if (options?.signal?.aborted)
        throw new DOMException("Aborted", "AbortError");

      await Promise.all(
        distinctDescriptorsToProcess.map(async (desc) => {
          try {
            const cleanPath = desc.filePath.split("?")[0].replace("/@fs", "");
            const tags = await exiftool.read(cleanPath);
            this.metadataCache![desc.filePath] = {
              mtime: desc.mtime,
              size: desc.size,
              hash: desc.hash,
              data: tags,
            };
            results[desc.filePath] = tags;
          } catch (err) {
            console.error(
              `[astro-image-pipeline] EXIF Extraction Failure for ${desc.filePath}:`,
              err,
            );
          }
        }),
      );

      await this.saveMetadataCache();
      return results;
    })();

    this.metadataLock = executionPromise;
    return await executionPromise;
  }

  public async getEmbeddings(
    filePaths: string[],
    options?: { signal?: AbortSignal },
  ): Promise<Record<string, number[]>> {
    const currentLock = this.embeddingLock.catch(() => {});
    const executionPromise = (async () => {
      await currentLock;
      if (options?.signal?.aborted)
        throw new DOMException("Aborted", "AbortError");

      const [, ...fileDescriptorsArray] = await Promise.all([
        this.embeddingCache ? Promise.resolve() : this.loadEmbeddingCache(),
        ...filePaths.map(async (filePath) => {
          const stats = await this.getFileStatsAndHash(filePath);
          return { filePath, ...stats };
        }),
      ]);

      if (!this.embeddingCache)
        throw Error(`[astro-image-pipeline] Failed to load embedding cache.`);

      const results: Record<string, number[]> = {};
      const distinctFilePaths = new Set();
      const distinctDescriptorsToProcess: typeof fileDescriptorsArray = [];
      let cacheHits = 0;

      for (const desc of fileDescriptorsArray) {
        const cached = this.embeddingCache[desc.filePath];
        if (
          cached &&
          cached.mtime === desc.mtime &&
          cached.size === desc.size &&
          cached.hash === desc.hash
        ) {
          results[desc.filePath] = cached.data;
          cacheHits++;
        } else {
          if (!distinctFilePaths.has(desc.filePath)) {
            distinctDescriptorsToProcess.push(desc);
            distinctFilePaths.add(desc.filePath);
          }
        }
      }

      console.log(
        `[astro-image-pipeline] Embedding cache hits: ${cacheHits} / ${filePaths.length}.`,
      );

      if (distinctDescriptorsToProcess.length === 0) return results;

      const extractor = await this.getPipeline();
      const totalBatches = Math.ceil(
        distinctDescriptorsToProcess.length / this.ML_BATCH_SIZE,
      );

      for (
        let i = 0;
        i < distinctDescriptorsToProcess.length;
        i += this.ML_BATCH_SIZE
      ) {
        if (options?.signal?.aborted)
          throw new DOMException("Aborted", "AbortError");

        const currentBatch = distinctDescriptorsToProcess.slice(
          i,
          i + this.ML_BATCH_SIZE,
        );
        try {
          const imageBuffers = await Promise.all(
            currentBatch.map(async (desc) => {
              const cleanPath = desc.filePath.split("?")[0].replace("/@fs", "");
              const { data, info } = await sharp(cleanPath)
                .resize(224, 224, { fit: "fill" })
                .removeAlpha()
                .raw()
                .toBuffer({ resolveWithObject: true });

              return new RawImage(
                new Uint8Array(data),
                info.width,
                info.height,
                3,
              );
            }),
          );

          const outputs = await extractor(imageBuffers, {
            pooling: "mean",
            normalize: true,
          });
          const [, embeddingDim] = outputs.dims;
          const flatData = outputs.data;

          currentBatch.forEach((desc, index) => {
            const start = index * embeddingDim;
            const end = start + embeddingDim;
            const vector = Array.from(
              flatData.subarray(start, end),
            ) as number[];

            this.embeddingCache![desc.filePath] = {
              mtime: desc.mtime,
              size: desc.size,
              hash: desc.hash,
              data: vector,
            };
            results[desc.filePath] = vector;
          });

          console.log(
            `[astro-image-pipeline] Completed embedding generation for batch ${i / this.ML_BATCH_SIZE + 1} of ${totalBatches}.`,
          );
        } catch (err) {
          console.error(
            `[astro-image-pipeline] ML Inference Batch Failure:`,
            err,
          );
        }
      }

      await this.saveEmbeddingCache();
      return results;
    })();

    this.embeddingLock = executionPromise;
    return await executionPromise;
  }

  public async shutdown() {
    try {
      await exiftool.end();
    } catch (e) {}
    if (this.visionPipeline) {
      this.visionPipeline = null;
      globalPipelinePromise = null;
    }
  }
}

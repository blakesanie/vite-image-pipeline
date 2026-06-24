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

let globalPipelinePromise: Promise<any> | null = null;

export class ImagePipelineEngine {
  private metadataCachePath: string;
  private embeddingCachePath: string;
  private modelCachePath: string;
  private metadataCache: MetadataCacheSchema | null = null;
  private embeddingCache: EmbeddingCacheSchema | null = null;
  private modelName: string;
  private visionPipeline: any = null;
  private ML_BATCH_SIZE: number;

  private metadataLock: Promise<any> = Promise.resolve();
  private embeddingLock: Promise<any> = Promise.resolve();

  constructor(
    options: {
      modelName?: string;
      metadataCachePath?: string;
      embeddingCachePath?: string;
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
      const stat = await fs.stat(filePath);
      const size = stat.size;
      const mtime = stat.mtimeMs;

      const sampleSize = Math.min(size, 4096);
      const handle = await fs.open(filePath, "r");
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
      console.log(
        `[astro-image-pipeline] Metadata cache written to ${this.metadataCachePath} for ${Object.keys(this.metadataCache || {}).length} items.`,
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
      console.log(
        `[astro-image-pipeline] Embedding cache written to ${this.embeddingCachePath} for ${Object.keys(this.embeddingCache || {}).length} items.`,
      );
    } catch (err) {
      console.error(
        "[astro-image-pipeline] Failed to sync embedding cache:",
        err,
      );
    }
  }

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
          const absolutePath = path.resolve(path.join(process.cwd(), filePath));
          const stats = await this.getFileStatsAndHash(absolutePath);
          return { filePath, absolutePath, ...stats };
        }),
      ]);

      if (!this.metadataCache)
        throw Error(`[astro-image-pipeline] Failed to load metadata cache.`);

      const results: Record<string, Tags> = {};
      const distinctFilePaths = new Set();
      const distinctDescriptorsToProcess: typeof fileDescriptorsArray = [];
      let cacheHits = 0;

      // Evaluate entire cache block synchronously in one pass
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

      if (distinctDescriptorsToProcess.length === 0) {
        return results;
      }

      if (options?.signal?.aborted)
        throw new DOMException("Aborted", "AbortError");

      // Process all Exif extractions concurrently via Promise.all
      await Promise.all(
        distinctDescriptorsToProcess.map(async (desc) => {
          try {
            const tags = await exiftool.read(desc.absolutePath);
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

      // Bootstrapping cache loads and hashing files concurrently
      const [, ...fileDescriptorsArray] = await Promise.all([
        this.embeddingCache ? Promise.resolve() : this.loadEmbeddingCache(),
        ...filePaths.map(async (filePath) => {
          const absolutePath = path.resolve(path.join(process.cwd(), filePath));
          const stats = await this.getFileStatsAndHash(absolutePath);
          return { filePath, absolutePath, ...stats };
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

      if (distinctDescriptorsToProcess.length === 0) {
        return results;
      }

      const extractor = await this.getPipeline();
      const totalBatches = Math.ceil(
        distinctDescriptorsToProcess.length / this.ML_BATCH_SIZE,
      );

      // Model inference steps are kept sequential to protect memory/CPU limits,
      // but internal processing steps are highly parallelized.
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
          // Parallel processing of image transformations via Sharp
          const imageBuffers = await Promise.all(
            currentBatch.map(async (desc) => {
              const { data, info } = await sharp(desc.absolutePath)
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
    } catch (e) {
      // Quiet close
    }
    if (this.visionPipeline) {
      this.visionPipeline = null;
      globalPipelinePromise = null;
    }
  }
}

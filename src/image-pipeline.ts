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
        console.error(`[astro-image-pipeline] Failed to load metadata cache at ${this.metadataCachePath}.`, e);
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
        console.error(`[astro-image-pipeline] Failed to load embedding cache at ${this.embeddingCachePath}.`, e);
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

  private async getFileStatsAndHash(filePath: string): Promise<{ size: number; mtime: number; hash: string }> {
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
      console.error(`[astro-image-pipeline] Failed to generate stats and hash for ${filePath}:`, err);
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
      console.log(`[astro-image-pipeline] Metadata cache written to ${this.metadataCachePath} for ${Object.keys(this.metadataCache || {}).length} items.`);
    } catch (err) {
      console.error("[astro-image-pipeline] Failed to sync metadata cache:", err);
    }
  }

  private async saveEmbeddingCache() {
    try {
      await fs.mkdir(path.dirname(this.embeddingCachePath), { recursive: true });
      await fs.writeFile(
        this.embeddingCachePath,
        JSON.stringify(this.embeddingCache, null, 2),
        "utf-8",
      );
      console.log(`[astro-image-pipeline] Embedding cache written to ${this.embeddingCachePath} for ${Object.keys(this.embeddingCache || {}).length} items.`);
    } catch (err) {
      console.error("[astro-image-pipeline] Failed to sync embedding cache:", err);
    }
  }

  public async getMetadata(
    filePaths: string[],
    options?: { signal?: AbortSignal },
  ): Promise<Record<string, Tags>> {
    const currentLock = this.metadataLock.catch(() => { });
    const executionPromise = (async () => {
      await currentLock;
      if (options?.signal?.aborted) throw new DOMException("Aborted", "AbortError");

      if (!this.metadataCache) await this.loadMetadataCache();
      if (!this.metadataCache) throw Error(`[astro-image-pipeline] Failed to load metadata cache.`);

      const results: Record<string, Tags> = {};
      const distinctFilePathsToProcess = new Set<string>();
      let cacheHits = 0;

      // Map out metadata descriptors up front safely
      const fileDescriptors: Record<string, { absolutePath: string; size: number; mtime: number; hash: string }> = {};
      await Promise.all(
        filePaths.map(async (filePath) => {
          const absolutePath = path.resolve(path.join(process.cwd(), filePath));
          const stats = await this.getFileStatsAndHash(absolutePath);
          fileDescriptors[filePath] = { absolutePath, ...stats };
        })
      );

      for (const filePath of filePaths) {
        const cached = this.metadataCache[filePath];
        const current = fileDescriptors[filePath];

        if (
          cached &&
          cached.mtime === current.mtime &&
          cached.size === current.size &&
          cached.hash === current.hash
        ) {
          results[filePath] = cached.data;
          cacheHits++;
        } else {
          distinctFilePathsToProcess.add(filePath);
        }
      }

      console.log("[astro-image-pipeline] Metadata cache hits:", cacheHits, "/", filePaths.length);

      if (distinctFilePathsToProcess.size === 0) {
        return results;
      }

      let processedSinceLastSave = 0;
      const FLUSH_INTERVAL = 10;

      for (const filePath of distinctFilePathsToProcess) {
        if (options?.signal?.aborted) {
          if (processedSinceLastSave > 0) await this.saveMetadataCache();
          throw new DOMException("Aborted", "AbortError");
        }

        try {
          const descriptor = fileDescriptors[filePath];
          const tags = await exiftool.read(descriptor.absolutePath);

          this.metadataCache[filePath] = {
            mtime: descriptor.mtime,
            size: descriptor.size,
            hash: descriptor.hash,
            data: tags
          };

          results[filePath] = tags;
          processedSinceLastSave++;

          if (processedSinceLastSave >= FLUSH_INTERVAL) {
            await this.saveMetadataCache();
            processedSinceLastSave = 0;
          }
        } catch (err) {
          console.error(`[astro-image-pipeline] EXIF Extraction Failure:`, err);
        }
      }

      if (processedSinceLastSave > 0) await this.saveMetadataCache();
      return results;
    })();

    this.metadataLock = executionPromise;
    return await executionPromise;
  }

  public async getEmbeddings(
    filePaths: string[],
    options?: { signal?: AbortSignal },
  ): Promise<Record<string, number[]>> {
    const currentLock = this.embeddingLock.catch(() => { });
    const executionPromise = (async () => {
      await currentLock;
      if (options?.signal?.aborted) throw new DOMException("Aborted", "AbortError");

      if (!this.embeddingCache) await this.loadEmbeddingCache();
      if (!this.embeddingCache) throw Error(`[astro-image-pipeline] Failed to load embedding cache.`);

      const results: Record<string, number[]> = {};
      const distinctFilePathsToProcess = new Set<string>();
      let cacheHits = 0;

      // Gather current system states for processing up front
      const fileDescriptors: Record<string, { absolutePath: string; size: number; mtime: number; hash: string }> = {};
      await Promise.all(
        filePaths.map(async (filePath) => {
          const absolutePath = path.resolve(path.join(process.cwd(), filePath));
          const stats = await this.getFileStatsAndHash(absolutePath);
          fileDescriptors[filePath] = { absolutePath, ...stats };
        })
      );

      for (const filePath of filePaths) {
        const cached = this.embeddingCache[filePath];
        const current = fileDescriptors[filePath];

        if (
          cached &&
          cached.mtime === current.mtime &&
          cached.size === current.size &&
          cached.hash === current.hash
        ) {
          results[filePath] = cached.data;
          cacheHits++;
        } else {
          distinctFilePathsToProcess.add(filePath);
        }
      }

      console.log(`[astro-image-pipeline] Embedding cache hits: ${cacheHits} / ${filePaths.length}.`);

      const pendingFilePaths = Array.from(distinctFilePathsToProcess);
      if (pendingFilePaths.length === 0) {
        return results;
      }

      const extractor = await this.getPipeline();
      const totalBatches = Math.ceil(pendingFilePaths.length / this.ML_BATCH_SIZE);

      for (let i = 0; i < pendingFilePaths.length; i += this.ML_BATCH_SIZE) {
        if (options?.signal?.aborted) {
          throw new DOMException("Aborted", "AbortError");
        }

        const currentBatchFilePaths = pendingFilePaths.slice(i, i + this.ML_BATCH_SIZE);
        try {
          const imageBuffers = await Promise.all(
            currentBatchFilePaths.map(async (filePath) => {
              const descriptor = fileDescriptors[filePath];
              const { data, info } = await sharp(descriptor.absolutePath)
                .resize(224, 224, { fit: "fill" })
                .removeAlpha()
                .raw()
                .toBuffer({ resolveWithObject: true });

              return new RawImage(new Uint8Array(data), info.width, info.height, 3);
            }),
          );

          const outputs = await extractor(imageBuffers, {
            pooling: "mean",
            normalize: true,
          });

          const [batchSize, embeddingDim] = outputs.dims;
          const flatData = outputs.data;

          currentBatchFilePaths.forEach((filePath, index) => {
            const start = index * embeddingDim;
            const end = start + embeddingDim;
            const vector = Array.from(flatData.subarray(start, end)) as number[];

            const descriptor = fileDescriptors[filePath];
            if (!this.embeddingCache) throw Error(`[astro-image-pipeline] Embedding cache not initialized.`);

            this.embeddingCache[filePath] = {
              mtime: descriptor.mtime,
              size: descriptor.size,
              hash: descriptor.hash,
              data: vector
            };
            results[filePath] = vector;
          });

          console.log(`[astro-image-pipeline] Completed embedding generation for batch ${i / this.ML_BATCH_SIZE + 1} of ${totalBatches}.`);

          await this.saveEmbeddingCache();
        } catch (err) {
          console.error(`[astro-image-pipeline] ML Inference Batch Failure:`, err);
        }
      }
      return results;
    })();

    this.embeddingLock = executionPromise;
    return await executionPromise;
  }

  public async shutdown() {
    try {
      await exiftool.end();
    } catch (e) {
      // Catch quiet drop
    }
    if (this.visionPipeline) {
      this.visionPipeline = null;
      globalPipelinePromise = null;
    }
  }
}
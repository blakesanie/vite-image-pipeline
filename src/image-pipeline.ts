// image-pipeline.ts
import crypto from "crypto";
import { exiftool, type Tags } from "exiftool-vendored";
import { pipeline, env, RawImage } from "@huggingface/transformers";
import fs from "fs/promises";
import { existsSync } from "fs";
import path from "path";
import sharp from "sharp";

export interface MetadataCacheSchema {
  [fileSha: string]: { [key: string]: any };
}

export interface EmbeddingCacheSchema {
  [fileSha: string]: number[];
}

let globalPipelinePromise: Promise<any> | null = null;

export class ImagePipelineEngine {
  private metadataCachePath: string;
  private embeddingCachePath: string;
  private modelCachePath: string;
  private metadataCache: MetadataCacheSchema = {};
  private embeddingCache: EmbeddingCacheSchema = {};
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

  public async loadCache() {
    if (existsSync(this.metadataCachePath)) {
      try {
        const raw = await fs.readFile(this.metadataCachePath, "utf-8");
        this.metadataCache = JSON.parse(raw);
      } catch (e) {
        this.metadataCache = {};
      }
    }
    if (existsSync(this.embeddingCachePath)) {
      try {
        const raw = await fs.readFile(this.embeddingCachePath, "utf-8");
        this.embeddingCache = JSON.parse(raw);
      } catch (e) {
        this.embeddingCache = {};
      }
    }
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
    console.log(`[astro-image-pipeline] Image embedding pipeline loaded.`)
    return this.visionPipeline;
  }

  private async getFileHash(filePath: string): Promise<string> {
    try {
      const stat = await fs.stat(filePath);

      // 1. Get size and modified time (mtime Ms)
      const size = stat.size;
      const mtime = stat.mtimeMs;

      // 2. Sample the first 4KB of the file to catch quick internal edits
      const sampleSize = Math.min(size, 4096);
      const handle = await fs.open(filePath, "r");
      const buffer = Buffer.alloc(sampleSize);

      await handle.read(buffer, 0, sampleSize, 0);
      await handle.close();

      // 3. Create a composite key using a fast algorithm (xxhash or md5)
      // We use md5 here purely as a fast non-cryptographic checksum
      const sampleHash = crypto.createHash("md5").update(buffer).digest("hex");

      return `${size}_${mtime}_${sampleHash}`;
    } catch (err) {
      console.error(`[astro-image-pipeline] Failed to generate fast hash for ${filePath}:`, err);
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
      console.log(`[astro-image-pipeline] Metadata cache written to ${this.metadataCachePath} for ${Object.keys(this.metadataCache).length} items.`);
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
      console.log(`[astro-image-pipeline] Embedding cache written to ${this.embeddingCachePath} for ${Object.keys(this.embeddingCache).length} items.`);
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
    const currentLock = this.metadataLock.catch(() => { });
    const executionPromise = (async () => {
      await currentLock;
      if (options?.signal?.aborted)
        throw new DOMException("Aborted", "AbortError");

      const startTime = Date.now();
      if (Object.keys(this.metadataCache || {}).length === 0) await this.loadCache();

      const results: Record<string, Tags> = {};
      const distinctFilePathsToProcess = new Set<string>();
      let cacheHits = 0;

      const absolutePaths: Record<string, string> = {};
      for (const filePath of filePaths) {
        absolutePaths[filePath] = path.resolve(path.join(process.cwd(), filePath));
      }

      const filePathToKey = new Map()
      await Promise.all(filePaths.map((async (filePath) => {
        const absolutePath = absolutePaths[filePath];
        const key = `${filePath}#${await this.getFileHash(absolutePath)}`;
        filePathToKey.set(filePath, key);
      })))

      for (const filePath of filePaths) {
        const key = filePathToKey.get(filePath);
        const cached = this.metadataCache[key];
        if (cached) {
          results[filePath] = cached;
          cacheHits++;
          continue;
        }
        distinctFilePathsToProcess.add(filePath);
      }

      console.log("Metadata cache hits:", cacheHits, "/", filePaths.length);

      if (distinctFilePathsToProcess.size === 0) {
        return results;
      }

      let processedSinceLastSave = 0;
      const FLUSH_INTERVAL = 10;

      for (const filePath of distinctFilePathsToProcess) {
        // 🛑 Stop immediately if client aborted connection
        if (options?.signal?.aborted) {
          if (processedSinceLastSave > 0) await this.saveMetadataCache();
          throw new DOMException("Aborted", "AbortError");
        }

        try {
          const absolutePath = absolutePaths[filePath];
          const tags = await exiftool.read(absolutePath);
          const key = filePathToKey.get(filePath);
          this.metadataCache[key] = tags;
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
    const results = await executionPromise;
    return results;
  }

  // private async processFileHashesBatch(
  //   filePaths: string[],
  //   cacheCheckFn: (sha: string) => any,
  //   onCacheHit: (absolutePath: string, cachedData: any) => void,
  //   onCacheMiss: (sha: string, absolutePath: string) => void,
  //   batchSize = 1,
  // ): Promise<void> {
  //   const processedPaths = new Set<string>();
  //   let n = Math.ceil(filePaths.length / batchSize);
  //   for (let i = 0; i < filePaths.length; i += batchSize) {
  //     const batchPaths = filePaths.slice(i, i + batchSize);
  //     await Promise.all(
  //       batchPaths.map(async (rawPath) => {
  //         if (processedPaths.has(rawPath)) return;
  //         processedPaths.add(rawPath);
  //         const absolutePath = path.resolve(rawPath);
  //         if (!existsSync(absolutePath)) return;
  //         try {
  //           const sha = await this.getFileHash(absolutePath);
  //           const cachedItem = cacheCheckFn(sha);
  //           if (cachedItem) onCacheHit(absolutePath, cachedItem);
  //           else onCacheMiss(sha, absolutePath);
  //         } catch (hashErr) {
  //           console.error(
  //             `[astro-image-pipeline] Failed hashing target ${rawPath}:`,
  //             hashErr,
  //           );
  //         }
  //       }),
  //     );
  //     console.log("Finished SHA calculation batch", i / batchSize + 1, "of", n);
  //   }
  // }

  public async getEmbeddings(
    filePaths: string[],
    options?: { signal?: AbortSignal },
  ): Promise<Record<string, number[]>> {
    const currentLock = this.embeddingLock.catch(() => { });
    const executionPromise = (async () => {
      await currentLock;
      if (options?.signal?.aborted)
        throw new DOMException("Aborted", "AbortError");

      const startTime = Date.now();
      await this.loadCache();

      const results: Record<string, number[]> = {};
      const distinctFilePathsToProcess = new Set<string>();
      let cacheHits = 0;

      const absolutePaths: Record<string, string> = {};
      for (const filePath of filePaths) {
        absolutePaths[filePath] = path.resolve(path.join(process.cwd(), filePath));
      }

      const filePathsToKey: Record<string, string> = {};

      await Promise.all(filePaths.map(async (filePath) => {
        const absolutePath = absolutePaths[filePath];
        const key = `${filePath}#${await this.getFileHash(absolutePath)}`;
        filePathsToKey[filePath] = key;
      }))

      for (const filePath of filePaths) {
        const key = filePathsToKey[filePath];
        const cached = this.embeddingCache[key];
        if (cached) {
          results[filePath] = cached;
          cacheHits++;
          continue;
        }
        distinctFilePathsToProcess.add(filePath);
      }

      console.log(`[astro-image-pipeline] Embedding cache hits: ${cacheHits} / ${filePaths.length}.`);

      const pendingFilePaths = Array.from(distinctFilePathsToProcess);
      if (pendingFilePaths.length === 0) {
        return results;
      }

      const extractor = await this.getPipeline();
      const totalBatches = Math.ceil(pendingFilePaths.length / this.ML_BATCH_SIZE);

      for (let i = 0; i < pendingFilePaths.length; i += this.ML_BATCH_SIZE) {
        // 🛑 Stop inference batches if client closed connection
        if (options?.signal?.aborted) {
          throw new DOMException("Aborted", "AbortError");
        }

        const currentBatchFilePaths = pendingFilePaths.slice(
          i,
          i + this.ML_BATCH_SIZE,
        );
        try {
          const imageBuffers = await Promise.all(
            currentBatchFilePaths.map(async (filePath) => {
              const absolutePath = absolutePaths[filePath];
              const { data, info } = await sharp(absolutePath)
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

          const vectors = outputs.map((output: any) => Array.from(output.data) as number[]);
          currentBatchFilePaths.forEach((filePath, index) => {
            const vector = vectors[index];
            const key = filePathsToKey[filePath];
            this.embeddingCache[key] = vector;
            results[filePath] = vector;
          });

          console.log(`[astro-image-pipeline] Completed embedding generation for batch ${i / this.ML_BATCH_SIZE + 1} of ${totalBatches}.`)

          await this.saveEmbeddingCache();
        } catch (err) {
          console.error(
            `[astro-image-pipeline] ML Inference Batch Failure:`,
            err,
          );
        }
      }
      return results;
    })();

    this.embeddingLock = executionPromise;
    const results = await executionPromise;
    return results;
  }

  public async shutdown() {
    // 1. Terminate the background ExifTool process cleanly
    try {
      await exiftool.end();
    } catch (e) {
      // Prevent unhandled rejections during close steps
    }

    // 2. Kill the vision pipeline context references so the event loop can clear
    if (this.visionPipeline) {
      this.visionPipeline = null;
      globalPipelinePromise = null;
    }
  }
}

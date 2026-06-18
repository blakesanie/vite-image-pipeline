import crypto from "crypto";
import { exiftool, type Tags } from "exiftool-vendored";
import { pipeline, env, RawImage } from "@huggingface/transformers";
import fs from "fs/promises";
import { existsSync } from "fs";
import path from "path";
import sharp from "sharp";

export interface MetadataCacheSchema {
  [fileSha: string]: {
    [key: string]: any;
  };
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
    console.log("[astro-image-pipeline] Loading cache layers from disk...");
    if (existsSync(this.metadataCachePath)) {
      try {
        const raw = await fs.readFile(this.metadataCachePath, "utf-8");
        this.metadataCache = JSON.parse(raw);
        console.log(
          `[astro-image-pipeline] Metadata cache hydrated (${Object.keys(this.metadataCache).length} entries)`,
        );
      } catch (e) {
        console.warn(
          "[astro-image-pipeline] Metadata cache corrupt or unreadable, resetting.",
        );
        this.metadataCache = {};
      }
    }

    if (existsSync(this.embeddingCachePath)) {
      try {
        const raw = await fs.readFile(this.embeddingCachePath, "utf-8");
        this.embeddingCache = JSON.parse(raw);
        console.log(
          `[astro-image-pipeline] Embedding cache hydrated (${Object.keys(this.embeddingCache).length} entries)`,
        );
      } catch (e) {
        console.warn(
          "[astro-image-pipeline] Embedding cache corrupt or unreadable, resetting.",
        );
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
          env.backends.onnx.wasm.numThreads = 4;
        }

        const modelFilePath = path.join(
          env.cacheDir,
          this.modelName,
          "onnx",
          "model.onnx",
        );

        const pipelineOptions: Record<string, any> = { device: "cpu" };
        let loadedPipeline: any = null;

        try {
          console.log(
            `[astro-image-pipeline] Searching for local model cache layout...`,
          );
          env.allowLocalModels = true;

          loadedPipeline = await pipeline(
            "image-feature-extraction",
            this.modelName,
            pipelineOptions,
          );

          console.log(
            `[astro-image-pipeline] Verified and active: ${modelFilePath}`,
          );
        } catch (localError) {
          console.log(
            `[astro-image-pipeline] Local model footprint missing/stale. Initiating network bootstrap...`,
          );
          env.allowLocalModels = false;

          loadedPipeline = await pipeline(
            "image-feature-extraction",
            this.modelName,
            pipelineOptions,
          );

          console.log(
            `[astro-image-pipeline] Download sequence complete. Saved natively under: ${env.cacheDir}`,
          );
        }

        return loadedPipeline;
      })();
    }

    this.visionPipeline = await globalPipelinePromise;
    return this.visionPipeline;
  }

  /**
   * Stream-based content hashing keeps memory low by avoiding massive buffer instantiations.
   */
  private async getFileHash(filePath: string): Promise<string> {
    const handle = await fs.open(filePath, "r");
    const stream = handle.createReadStream();
    const hash = crypto.createHash("sha256");

    return new Promise((resolve, reject) => {
      stream.on("data", (chunk) => hash.update(chunk));
      stream.on("end", () => {
        handle.close().catch(console.error);
        resolve(hash.digest("hex"));
      });
      stream.on("error", (err) => {
        handle.close().catch(console.error);
        reject(err);
      });
    });
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
        `[astro-image-pipeline] Metadata cache flushed sequentially (${Object.keys(this.metadataCache).length} keys total)`,
      );
    } catch (err) {
      console.error(
        "[astro-image-pipeline] Failed to sync metadata cache to disk:",
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
        `[astro-image-pipeline] Embedding cache flushed sequentially (${Object.keys(this.embeddingCache).length} keys total)`,
      );
    } catch (err) {
      console.error(
        "[astro-image-pipeline] Failed to sync embedding cache to disk:",
        err,
      );
    }
  }

  /**
   * 📸 Public Metadata Extraction Method (Content-Addressable)
   */
  public async getMetadata(filePaths: string[]): Promise<Record<string, Tags>> {
    const startTime = Date.now();
    await this.loadCache();

    const results: Record<string, Tags> = {};
    const distinctHashesToProcess = new Set<string>();
    const hashToPathsMap: Record<string, string> = {};
    const processedPaths = new Set<string>();
    let cacheHits = 0;

    console.log(
      `[astro-image-pipeline] Mapping content hashes for ${filePaths.length} incoming targets`,
    );

    await this.processFileHashesBatch(
      filePaths,
      (sha) => this.metadataCache[sha],
      (absolutePath, cachedTags) => {
        results[absolutePath] = cachedTags;
        cacheHits++;
      },
      (sha, absolutePath) => {
        distinctHashesToProcess.add(sha);
        hashToPathsMap[sha] = absolutePath;
      },
    );

    console.log(
      `[astro-image-pipeline] Metadata Manifest -> Hits: ${cacheHits}, Read-Queue: ${distinctHashesToProcess.size}`,
    );
    if (distinctHashesToProcess.size === 0) return results;

    let processedSinceLastSave = 0;
    // Periodic write interval bounds execution safety
    const FLUSH_INTERVAL = 10;

    for (const sha of distinctHashesToProcess) {
      const targetPath = hashToPathsMap[sha];
      try {
        const tags = await exiftool.read(targetPath);
        this.metadataCache[sha] = tags;
        results[targetPath] = tags;

        processedSinceLastSave++;
        if (processedSinceLastSave >= FLUSH_INTERVAL) {
          console.log(
            `[astro-image-pipeline] Intermediate metadata threshold hit (${processedSinceLastSave} items). Syncing checkpoint...`,
          );
          await this.saveMetadataCache();
          processedSinceLastSave = 0;
        }
      } catch (err) {
        console.error(
          `[astro-image-pipeline] EXIF Extraction Failure for ${path.basename(targetPath)}:`,
          err,
        );
      }
    }

    if (processedSinceLastSave > 0) {
      await this.saveMetadataCache();
    }

    console.log(
      `[astro-image-pipeline] Metadata extraction run completed in ${((Date.now() - startTime) / 1000).toFixed(2)}s`,
    );
    return results;
  }

  private async processFileHashesBatch(
    filePaths: string[],
    cacheCheckFn: (sha: string) => any,
    onCacheHit: (absolutePath: string, cachedData: any) => void,
    onCacheMiss: (sha: string, absolutePath: string) => void,
    batchSize = 4,
  ): Promise<void> {
    const processedPaths = new Set<string>();
    const n = Math.ceil(filePaths.length / batchSize);

    for (let i = 0; i < filePaths.length; i += batchSize) {
      const batchPaths = filePaths.slice(i, i + batchSize);

      await Promise.all(
        batchPaths.map(async (rawPath) => {
          if (processedPaths.has(rawPath)) return;
          processedPaths.add(rawPath);

          const absolutePath = path.resolve(rawPath);
          if (!existsSync(absolutePath)) return;

          try {
            const sha = await this.getFileHash(absolutePath);
            const cachedItem = cacheCheckFn(sha);

            if (cachedItem) {
              onCacheHit(absolutePath, cachedItem);
            } else {
              onCacheMiss(sha, absolutePath);
            }
          } catch (hashErr) {
            console.error(
              `[astro-image-pipeline] Failed hashing target ${rawPath}:`,
              hashErr,
            );
          }
        }),
      );
      console.log("finished file hash batch", i / batchSize + 1, "of", n);
    }
  }

  /**
   * 🧠 Public WebGPU Embedding Extraction Method (Content-Addressable)
   */
  public async getEmbeddings(
    filePaths: string[],
  ): Promise<Record<string, number[]>> {
    const startTime = Date.now();
    await this.loadCache();

    const results: Record<string, number[]> = {};
    const distinctHashesToProcess = new Set<string>();
    const hashToPathsMap: Record<string, string> = {};
    const processedPaths = new Set<string>();
    let cacheHits = 0;

    console.log(
      `[astro-image-pipeline] Mapping content hashes for vector targets...`,
    );

    await this.processFileHashesBatch(
      filePaths,
      (sha) => this.embeddingCache[sha],
      (absolutePath, cachedVector) => {
        results[absolutePath] = cachedVector;
        cacheHits++;
      },
      (sha, absolutePath) => {
        distinctHashesToProcess.add(sha);
        hashToPathsMap[sha] = absolutePath;
      },
    );

    const pendingHashes = Array.from(distinctHashesToProcess);
    console.log(
      `[astro-image-pipeline] Embedding Manifest -> Hits: ${cacheHits}, Compute-Queue: ${pendingHashes.length}`,
    );

    if (pendingHashes.length === 0) return results;

    const extractor = await this.getPipeline();
    const totalBatches = Math.ceil(pendingHashes.length / this.ML_BATCH_SIZE);

    for (let i = 0; i < pendingHashes.length; i += this.ML_BATCH_SIZE) {
      const currentBatchHashes = pendingHashes.slice(i, i + this.ML_BATCH_SIZE);
      const currentBatchIndex = Math.floor(i / this.ML_BATCH_SIZE) + 1;

      console.log(
        `[astro-image-pipeline] Processing ML inference batch ${currentBatchIndex}/${totalBatches} (${currentBatchHashes.length} items)...`,
      );

      try {
        // Parallelized sharp processing throttled implicitly by the batch sizing constraints
        const imageBuffers = await Promise.all(
          currentBatchHashes.map(async (sha) => {
            const samplePath = hashToPathsMap[sha];
            const { data, info } = await sharp(samplePath)
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

        currentBatchHashes.forEach((sha, index) => {
          const vector = Array.from(outputs[index].data) as number[];
          this.embeddingCache[sha] = vector;
          results[hashToPathsMap[sha]] = vector;
        });

        // Safe checkpoint save: write cache after every single batch evaluation run succeeds
        console.log(
          `[astro-image-pipeline] Batch ${currentBatchIndex} evaluation finished cleanly. Saving progress...`,
        );
        await this.saveEmbeddingCache();
      } catch (err) {
        console.error(
          `[astro-image-pipeline] Critical execution fault in ML Inference Batch ${currentBatchIndex}:`,
          err,
        );
      }
    }

    console.log(
      `[astro-image-pipeline] Embedding generation run completed in ${((Date.now() - startTime) / 1000).toFixed(2)}s`,
    );
    return results;
  }

  public async shutdown() {
    console.log("[astro-image-pipeline] Halting active ExifTool subsystems...");
    await exiftool.end();
  }
}

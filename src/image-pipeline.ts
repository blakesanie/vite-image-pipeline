import crypto from "crypto";
import { exiftool, type Tags } from "exiftool-vendored";
import { pipeline, env, RawImage } from "@huggingface/transformers";
import fs from "fs/promises";
import { existsSync } from "fs";
import path from "path";
import sharp from "sharp";

export interface MetadataCacheSchema {
  // Keyed by SHA-256 content hash instead of system file path
  [fileSha: string]: {
    [key: string]: any;
  };
}

export interface EmbeddingCacheSchema {
  // Keyed by SHA-256 content hash instead of system file path
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
    console.log("start getPipeline()");
    if (this.visionPipeline) return this.visionPipeline;

    // Use a unified promise chain to catch execution blocks inside the same thread
    if (!globalPipelinePromise) {
      console.log("inside globalPipelinePromise");
      globalPipelinePromise = (async () => {
        console.log("inside globalPipelinePromise async");

        // 1. Configure standard global storage locations
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

        const pipelineOptions: Record<string, any> = {
          device: "cpu",
        };

        let loadedPipeline: any = null;

        // 2. Strategy: Try reading from local cache natively first
        try {
          console.log(
            `[astro-image-pipeline] Testing local storage engine for model cache structure...`,
          );

          // Force the framework to ONLY look locally. It throws if anything is missing/broken.
          env.allowLocalModels = true;

          loadedPipeline = await pipeline(
            "image-feature-extraction",
            this.modelName,
            pipelineOptions,
          );

          console.log(
            `[astro-image-pipeline] Model successfully verified and loaded from disk location: ${modelFilePath}`,
          );
        } catch (localError) {
          // 3. Fallback: Local lookup failed, flag a web-based download bootstrap sequence
          console.log(
            `[astro-image-pipeline] Model not found locally or structure corrupted. Bootstrapping initial download...`,
          );

          env.allowLocalModels = false; // Open network layer permission

          loadedPipeline = await pipeline(
            "image-feature-extraction",
            this.modelName,
            pipelineOptions,
          );

          console.log(
            `[astro-image-pipeline] Model downloaded and saved natively to directory structure under: ${env.cacheDir}`,
          );
        }

        return loadedPipeline;
      })();
    }

    this.visionPipeline = await globalPipelinePromise;
    return this.visionPipeline;
  }

  private async getFileHash(filePath: string): Promise<string> {
    const fileBuffer = await fs.readFile(filePath);
    return crypto.createHash("sha256").update(fileBuffer).digest("hex");
  }

  private async saveCache() {
    await fs.mkdir(path.dirname(this.metadataCachePath), { recursive: true });
    await fs.writeFile(
      this.metadataCachePath,
      JSON.stringify(this.metadataCache, null, 2),
      "utf-8",
    );
    console.log(
      `[astro-image-pipeline] Cache saved to ${this.metadataCachePath}`,
    );

    await fs.mkdir(path.dirname(this.embeddingCachePath), { recursive: true });
    await fs.writeFile(
      this.embeddingCachePath,
      JSON.stringify(this.embeddingCache, null, 2),
      "utf-8",
    );
    console.log(
      `[astro-image-pipeline] Cache saved to ${this.embeddingCachePath}`,
    );
  }

  /**
   * 📸 Public Metadata Extraction Method (Content-Addressable)
   */
  public async getMetadata(filePaths: string[]): Promise<Record<string, Tags>> {
    await this.loadCache();
    const results: Record<string, Tags> = {};
    // Unique hashes that actually require an ExifTool disk read
    const distinctHashesToProcess = new Set<string>();
    // Maps a hash back to an array of paths that share it (for unpacking results)
    const hashToPathsMap: Record<string, string> = {};

    const rawPathsConsidered = new Set<string>();
    let cacheHits = 0;

    // 1. Calculate hashes concurrently to evaluate cache hits
    for (const rawPath of filePaths) {
      if (rawPathsConsidered.has(rawPath)) continue;
      rawPathsConsidered.add(rawPath);
      const absolutePath = path.resolve(rawPath);
      if (!existsSync(absolutePath)) continue;

      const sha = await this.getFileHash(absolutePath);

      if (this.metadataCache[sha]) {
        results[absolutePath] = this.metadataCache[sha];
        cacheHits++;
        continue;
      }

      distinctHashesToProcess.add(sha);
      hashToPathsMap[sha] = absolutePath;
    }

    console.log(
      `[astro-image-pipeline] Metadata statistics -> Cached items utilized: ${cacheHits}, Items to parse: ${distinctHashesToProcess.size}`,
    );

    console.log("distinctHashesToProcess", distinctHashesToProcess);
    if (distinctHashesToProcess.size === 0) return results;

    // 2. Process metadata for hashes not found in the cache

    for (const sha of distinctHashesToProcess) {
      // Pick the first available path matching this hash to pull metadata from
      const targetPath = hashToPathsMap[sha];
      const absoluteTargetPath = path.resolve(targetPath);

      try {
        const tags = await exiftool.read(absoluteTargetPath);

        // Initialize or update the cache structural entry for this SHA
        this.metadataCache[sha] = tags;

        // Distribute the tags to all input paths sharing this SHA
        results[hashToPathsMap[sha]] = tags;
      } catch (err) {
        console.error(
          `[astro-image-pipeline] EXIF Error for hash ${sha.slice(0, 8)} (${path.basename(absoluteTargetPath)}):`,
          err,
        );
      }
    }

    await this.saveCache();
    return results;
  }

  /**
   * 🧠 Public WebGPU Embedding Extraction Method (Content-Addressable)
   */
  public async getEmbeddings(
    filePaths: string[],
  ): Promise<Record<string, number[]>> {
    await this.loadCache();
    const results: Record<string, number[]> = {};

    const distinctHashesToProcess = new Set<string>();
    const hashToPathsMap: Record<string, string> = {};

    const rawPathsConsidered = new Set<string>();
    let cacheHits = 0;

    // 1. Calculate hashes concurrently to evaluate cache hits
    for (const rawPath of filePaths) {
      if (rawPathsConsidered.has(rawPath)) continue;
      rawPathsConsidered.add(rawPath);
      const absolutePath = path.resolve(rawPath);
      if (!existsSync(absolutePath)) continue;

      const sha = await this.getFileHash(absolutePath);

      if (this.embeddingCache[sha]) {
        results[absolutePath] = this.embeddingCache[sha];
        cacheHits++;
        continue;
      }
      distinctHashesToProcess.add(sha);
      hashToPathsMap[sha] = absolutePath;
    }

    const pendingHashes = Array.from(distinctHashesToProcess);
    console.log(
      `[astro-image-pipeline] Embedding statistics -> Cached items utilized: ${cacheHits}, Items to calculate: ${pendingHashes.length}`,
    );

    const _ = await this.getPipeline();
    if (pendingHashes.length === 0) return results;

    // 2. Batch Tensor WebGPU Evaluation Track
    const extractor = await this.getPipeline();

    for (let i = 0; i < pendingHashes.length; i += this.ML_BATCH_SIZE) {
      const currentBatchHashes = pendingHashes.slice(i, i + this.ML_BATCH_SIZE);

      try {
        const imageBuffers = await Promise.all(
          currentBatchHashes.map(async (sha) => {
            const samplePath = path.resolve(hashToPathsMap[sha]);

            // 1. Tell sharp to output raw, uncompressed RGB pixel data
            const { data, info } = await sharp(samplePath)
              .resize(224, 224, { fit: "fill" })
              .removeAlpha() // Ensures 3 channels (RGB) instead of 4 (RGBA)
              .raw()
              .toBuffer({ resolveWithObject: true });

            // 2. Instantiated RawImage manually using Uint8Array
            // This bypasses the buggy .read() signatures entirely
            return new RawImage(
              new Uint8Array(data),
              info.width,
              info.height,
              3, // 3 channels: RGB
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
      } catch (err) {
        console.error(`[astro-image-pipeline] Embedding Batch Error:`, err);
      }
    }

    await this.saveCache();
    return results;
  }

  public async shutdown() {
    await exiftool.end();
  }
}

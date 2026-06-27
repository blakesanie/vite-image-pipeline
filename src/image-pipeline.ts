// image-pipeline.ts
/// <reference types="vite/client" />

import crypto from "crypto";
import { exiftool, type Tags } from "exiftool-vendored";
import { pipeline, env, RawImage } from "@huggingface/transformers";
import fs from "fs/promises";
import { existsSync } from "fs";
import path from "path";
import sharp from "sharp";
import type { AstroIntegration } from "astro";
import { loadCache, saveCache, getFileStatsAndHash, resolveToAbsolutePath } from "./utils.js";
import { RemoteImageOptions, RemotePlatform, remotePlatformId } from "./remote.js";

interface CacheEntry<T> {
  mtime: number;
  size: number;
  hash: string;
  data: T;
}

export interface ImagePipelineOptions {
  modelName?: string;
  metadataCachePath?: string;
  embeddingCachePath?: string;
  colorCachePath?: string;
  blurCachePath?: string;
  batchSize?: number;
  modelCachePath?: string;
}

let options = {
  metadataCachePath: path.resolve(".image-pipeline/metadata-cache.json"),
  embeddingCachePath: path.resolve(".image-pipeline/embedding-cache.json"),
  colorCachePath: path.resolve(".image-pipeline/color-cache.json"),
  blurCachePath: path.resolve(".image-pipeline/blur-cache.json"),
  modelCachePath: path.resolve(".image-pipeline/models"),
  modelName: "Xenova/clip-vit-base-patch32",
  batchSize: 4,
};

export function setOptions(newOptions: ImagePipelineOptions) {
  options = {
    ...options,
    ...newOptions
  }
}

// metadata

declare global {
  var metadataCache: Record<string, CacheEntry<Tags>> | null;
  var metadataLock: Promise<any>;
}

globalThis.metadataCache = null;
globalThis.metadataLock = Promise.resolve();

export async function getMetadata(
  filePaths: string[],
): Promise<Record<string, Tags>> {
  const currentLock = globalThis.metadataLock.catch(() => { });
  const executionPromise = (async () => {
    await currentLock;

    if (!globalThis.metadataCache) {
      globalThis.metadataCache = await loadCache<CacheEntry<Tags>>(options.metadataCachePath);
    }
    const fileDescriptorsArray = await Promise.all([
      ...filePaths.map(async (filePath) => {
        const stats = await getFileStatsAndHash(filePath);
        return { filePath, ...stats };
      }),
    ]);

    if (!globalThis.metadataCache)
      throw Error(`[astro-image-pipeline] Failed to load metadata cache.`);

    const results: Record<string, Tags> = {};
    const distinctFilePaths = new Set();
    const distinctDescriptorsToProcess: typeof fileDescriptorsArray = [];
    let cacheHits = 0;

    for (const desc of fileDescriptorsArray) {
      const cached = globalThis.metadataCache![desc.filePath];
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
    await Promise.all(
      distinctDescriptorsToProcess.map(async (desc) => {
        try {
          const cleanPath = resolveToAbsolutePath(desc.filePath)
          const tags = await exiftool.read(cleanPath);
          metadataCache![desc.filePath] = {
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

    await saveCache(options.metadataCachePath, globalThis.metadataCache);
    return results;
  })();

  metadataLock = executionPromise;
  return await executionPromise;
}

// embeddings

declare global {
  var embeddingCache: Record<string, CacheEntry<number[]>> | null;
  var embeddingLock: Promise<any>;
  var visionPipeline: any;
}

globalThis.embeddingCache = null;
globalThis.embeddingLock = Promise.resolve();
globalThis.visionPipeline = null;

export async function getEmbeddings(
  filePaths: string[],
): Promise<Record<string, number[]>> {
  const currentLock = embeddingLock.catch(() => { });
  const executionPromise = (async () => {
    await currentLock;

    if (!globalThis.embeddingCache) {
      globalThis.embeddingCache = await loadCache<CacheEntry<number[]>>(options.embeddingCachePath);
    }

    const fileDescriptorsArray = await Promise.all([
      ...filePaths.map(async (filePath) => {
        const stats = await getFileStatsAndHash(filePath);
        return { filePath, ...stats };
      }),
    ]);

    if (!globalThis.embeddingCache)
      throw Error(`[astro-image-pipeline] Failed to load embedding cache.`);

    const results: Record<string, number[]> = {};
    const distinctFilePaths = new Set();
    const distinctDescriptorsToProcess: typeof fileDescriptorsArray = [];
    let cacheHits = 0;

    for (const desc of fileDescriptorsArray) {
      const cached = globalThis.embeddingCache![desc.filePath];
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

    const extractor = await getPipeline();
    const totalBatches = Math.ceil(
      distinctDescriptorsToProcess.length / options.batchSize,
    );

    for (
      let i = 0;
      i < distinctDescriptorsToProcess.length;
      i += options.batchSize
    ) {

      const currentBatch = distinctDescriptorsToProcess.slice(
        i,
        i + options.batchSize,
      );
      try {
        const imageBuffers = await Promise.all(
          currentBatch.map(async (desc) => {
            const cleanPath = resolveToAbsolutePath(desc.filePath)
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

          globalThis.embeddingCache![desc.filePath] = {
            mtime: desc.mtime,
            size: desc.size,
            hash: desc.hash,
            data: vector,
          };
          results[desc.filePath] = vector;
        });

        console.log(
          `[astro-image-pipeline] Completed embedding generation for batch ${i / options.batchSize + 1} of ${totalBatches}.`,
        );
      } catch (err) {
        console.error(
          `[astro-image-pipeline] ML Inference Batch Failure:`,
          err,
        );
      }
    }

    await saveCache(options.embeddingCachePath, globalThis.embeddingCache);
    return results;
  })();

  globalThis.embeddingLock = executionPromise;
  return await executionPromise;
}

async function getPipeline() {
  if (globalThis.visionPipeline) return globalThis.visionPipeline;

  env.useFSCache = true;
  env.cacheDir = path.resolve(options.modelCachePath);
  if (env.backends?.onnx?.wasm) {
    env.backends.onnx.wasm.numThreads =
      process.env.NODE_ENV === "production" ? 1 : 4;
  }
  env.allowLocalModels = true;
  try {
    globalThis.visionPipeline =
      await pipeline("image-feature-extraction", options.modelName, {
        device: "cpu",
      });
  } catch {
    env.allowLocalModels = false;
    globalThis.visionPipeline =
      await pipeline("image-feature-extraction", options.modelName, {
        device: "cpu",
      });
  }
  console.log(`[astro-image-pipeline] Image embedding pipeline loaded.`);
  return globalThis.visionPipeline;
}

// blur placeholder

declare global {
  var blurCache: Record<string, CacheEntry<string>> | null;
  var blurLock: Promise<any>;
}

globalThis.blurCache = null;
globalThis.blurLock = Promise.resolve();

export async function getImageBlurPlaceholders(
  filePaths: string[],
): Promise<Record<string, string>> {
  const currentLock = globalThis.blurLock.catch(() => { });
  const executionPromise = (async () => {
    await currentLock;

    if (!globalThis.blurCache) globalThis.blurCache = await loadCache(options.blurCachePath);

    const fileDescriptorsArray = await Promise.all([
      ...filePaths.map(async (filePath) => {
        const stats = await getFileStatsAndHash(filePath);
        return { filePath, ...stats };
      }),
    ]);

    if (!blurCache)
      throw Error(`[astro-image-pipeline] Failed to load blur cache.`);

    const results: Record<string, string> = {};
    const distinctDescriptorsToProcess: typeof fileDescriptorsArray = [];
    let cacheHits = 0;

    for (const desc of fileDescriptorsArray) {
      const cached = globalThis.blurCache![desc.filePath];
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

    await Promise.all(
      distinctDescriptorsToProcess.map(async (desc) => {
        try {
          const sharpInstance = sharp(
            resolveToAbsolutePath(desc.filePath),
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

          globalThis.blurCache![desc.filePath] = {
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

    await saveCache(options.blurCachePath, globalThis.blurCache);
    return results;
  })();

  globalThis.blurLock = executionPromise;
  return await executionPromise;
}

// placeholder color

declare global {
  var colorCache: Record<string, CacheEntry<{ r: number; g: number; b: number }>> | null;
  var colorLock: Promise<any>;
}

globalThis.colorCache = null;
globalThis.colorLock = Promise.resolve();

export async function getImageColors(
  filePaths: string[]
): Promise<Record<string, { r: number; g: number; b: number }>> {
  const currentLock = globalThis.colorLock.catch(() => { });
  const executionPromise = (async () => {
    await currentLock;

    if (!globalThis.colorCache) globalThis.colorCache = await loadCache(options.colorCachePath);

    const fileDescriptorsArray = await Promise.all([
      ...filePaths.map(async (filePath) => {
        const stats = await getFileStatsAndHash(filePath);
        return { filePath, ...stats };
      }),
    ]);

    if (!globalThis.colorCache)
      throw Error(`[astro-image-pipeline] Failed to load color cache.`);

    const results: Record<string, { r: number; g: number; b: number }> = {};
    const distinctDescriptorsToProcess: typeof fileDescriptorsArray = [];
    let cacheHits = 0;

    for (const desc of fileDescriptorsArray) {
      const cached = globalThis.colorCache![desc.filePath];
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

    await Promise.all(
      distinctDescriptorsToProcess.map(async (desc) => {
        try {
          const sharpInstance = sharp(
            resolveToAbsolutePath(desc.filePath),
          );
          const { dominant } = await sharpInstance.stats();
          const payload = { r: dominant.r, g: dominant.g, b: dominant.b };

          globalThis.colorCache![desc.filePath] = {
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
    await saveCache(options.colorCachePath, globalThis.colorCache);
    return results;
  })();

  globalThis.colorLock = executionPromise;
  return await executionPromise;
}

// remote image

declare global {
  var remotePlatforms: Map<string, RemotePlatform>;
  var registryLock: Promise<string[]>;
}
if (!globalThis.remotePlatforms) {
  globalThis.remotePlatforms = new Map<string, RemotePlatform>();
}
if (!globalThis.registryLock) {
  globalThis.registryLock = Promise.resolve([]);
}

export async function uploadRemoteImages(
  remoteOptions: RemoteImageOptions,
  filepaths: string[],
  remoteCondition: () => boolean = () => (import.meta as any).env?.PROD
): Promise<string[]> {

  const shouldRemote = remoteCondition();
  if (!shouldRemote) {
    return filepaths;
  }

  console.log("[astro-image-pipeline] should upload to remote");

  const currentLock = globalThis.registryLock.catch(() => { });

  const executionPromise = (async () => {
    await currentLock;

    const id = remotePlatformId(remoteOptions);
    console.log("[astro-image-pipeline] remote platform id", id);
    let platform = globalThis.remotePlatforms.get(id);
    if (!platform) {
      platform = RemotePlatform(remoteOptions);
      platform.validate();
      globalThis.remotePlatforms.set(id, platform);
    }

    return filepaths.map(filepath => {
      const cleanPath = filepath.split("?")[0];
      platform.filepaths.add(cleanPath);
      return platform.generateRemoteUrl(cleanPath);
    })
  })();
  globalThis.registryLock = executionPromise;
  return await executionPromise;
}

let uploadLock: Promise<void> = Promise.resolve();

async function processRemoteUploads() {
  console.log(`[astro-image-pipeline] Starting to upload remote images`, globalThis.remotePlatforms)
  const currentLock = uploadLock.catch(() => { });
  const executionPromise = (async () => {
    await currentLock;
    for (const [id, platform] of globalThis.remotePlatforms.entries()) {
      try {
        console.log(`[astro-image-pipeline] Uploading remote images for platform ${id}`);
        await platform.upload();
        console.log(`[astro-image-pipeline] Uploaded remote images for platform ${id}`);
      } catch (e) {
        console.error(`[astro-image-pipeline] Failed to upload remote images for platform ${id}:`, e);
      }
      globalThis.remotePlatforms.delete(id);
    }
  })();
  uploadLock = executionPromise;
  await executionPromise;
}


async function stop() {
  try {
    await exiftool.end();
  } catch (e) { }
  if (globalThis.visionPipeline) {
    globalThis.visionPipeline = null;
  }
}

export function astroImagePipelinePlugin(): AstroIntegration {
  return {
    name: "image-pipeline",
    hooks: {
      "astro:build:generated": stop,
      "astro:build:done": processRemoteUploads,
    },
  };
}
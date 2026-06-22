import type { AstroIntegration } from "astro";
import path from "path";
import { ImagePipelineEngine } from "../image-pipeline.js";

let globalPipelineInstance: ImagePipelineEngine | null = null;
let savedPluginOptions: AstroPluginOptions | undefined = undefined;

interface AstroPluginOptions {
  modelName?: string;
  batchSize?: number;
}

interface RunOptions {
  signal?: AbortSignal;
}

const getEngine = () => {
  if (!globalPipelineInstance) {
    globalPipelineInstance = new ImagePipelineEngine({
      modelName: savedPluginOptions?.modelName,
      metadataCachePath: path.resolve(
        ".astro/astro-image-pipeline/metadata-cache.json",
      ),
      embeddingCachePath: path.resolve(
        ".astro/astro-image-pipeline/embedding-cache.json",
      ),
      batchSize: savedPluginOptions?.batchSize,
      modelCachePath: path.resolve(".astro/astro-image-pipeline/models"),
    });
  }
  return globalPipelineInstance;
};

// Lazily initialized when called by pages
export async function getImageMetadata(
  filePaths: string[],
  options?: RunOptions,
) {
  return getEngine().getMetadata(filePaths, options);
}

// Lazily initialized when called by pages
export async function getImageEmbeddings(
  filePaths: string[],
  options?: RunOptions,
) {
  return getEngine().getEmbeddings(filePaths, options);
}

export default function imagePipeline(
  options?: AstroPluginOptions,
): AstroIntegration {
  return {
    name: "image-pipeline",
    hooks: {
      "astro:config:setup": () => {
        // Save the config options for later use, but DO NOT initialize the engine yet
        savedPluginOptions = options;
      },
      "astro:build:generated": async () => {
        if (globalPipelineInstance) {
          await globalPipelineInstance.shutdown();
          globalPipelineInstance = null;
        }
      },
      "astro:build:done": async () => {
        if (globalPipelineInstance) {
          await globalPipelineInstance.shutdown();
          globalPipelineInstance = null;
        }
      },
      "astro:server:done": async () => {
        if (globalPipelineInstance) {
          await globalPipelineInstance.shutdown();
          globalPipelineInstance = null;
        }
      },
    },
  };
}

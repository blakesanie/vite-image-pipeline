import type { AstroIntegration } from 'astro';
import path from 'path';
import { ImagePipelineEngine } from '../image-pipeline.js';

let globalPipelineInstance: ImagePipelineEngine | null = null;

interface AstroPluginOptions {
    modelName?: string; batchSize?: number
}

const getEngine = (options?: AstroPluginOptions) => {
    if (!globalPipelineInstance) {
        globalPipelineInstance = new ImagePipelineEngine({
            modelName: options?.modelName,
            metadataCachePath: path.resolve('.astro/astro-image-pipeline/metadata-cache.json'),
            embeddingCachePath: path.resolve('.astro/astro-image-pipeline/embedding-cache.json'),
            batchSize: options?.batchSize,
            modelCachePath: path.resolve('.astro/astro-image-pipeline/models')
        });
    }
    return globalPipelineInstance;
};

export async function getImageMetadata(filePaths: string[]) {
    return getEngine().getMetadata(filePaths);
}

export async function getImageEmbeddings(filePaths: string[]) {
    return getEngine().getEmbeddings(filePaths);
}

export default function imagePipeline(options?: AstroPluginOptions): AstroIntegration {
    return {
        name: 'image-pipeline',
        hooks: {
            'astro:config:setup': () => {
                globalPipelineInstance = getEngine(options);
            },
            'astro:build:done': async () => { if (globalPipelineInstance) await globalPipelineInstance.shutdown(); },
            'astro:server:done': async () => { if (globalPipelineInstance) await globalPipelineInstance.shutdown(); }
        },
    };
}
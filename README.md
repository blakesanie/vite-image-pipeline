# vite-image-pipeline

A high-performance, asset-caching image pipeline designed for [**Vite**](https://vite.dev/ "null")\-based projects ([**Astro**](https://astro.build/ "null"), [**React**](https://react.dev/ "null"), [**Svelte**](https://svelte.dev/ "null"), [**Vue**](https://vuejs.org/ "null"), [**Solid**](https://www.solidjs.com/ "null"), and more). It extracts rich metadata, generates ML-driven vector embeddings (using local CLIP models), computes blur placeholders, samples dominant colors, and handles remote platform uploads during production builds.

While `vite-image-pipeline` is completely framework-agnostic and can be utilized in any Vite context, it also includes a dedicated integration for [**Astro**](https://astro.build/ "null") to automatically coordinate cleanup and remote synchronization on production builds.

This package is designed to work seamlessly **in conjunction with** image compression and optimization tools such as [**vite-imagetools**](https://github.com/JonasKruckenberg/imagetools "null") or Astro's native [**Image Component**](https://docs.astro.build/en/guides/images/ "null"). It processes your original source assets to power features like semantic visual search, UI color-matching, and custom lazy-loading states, leaving the final layout rendering and resizing to your preferred UI layer.

## Features

-   **Aggressive Content-Based Caching:** Uses file modification times, sizes, and cryptographic hashes to ensure images are processed exactly once unless modified.
    
-   **EXIF Metadata Extraction:** Powered by [**exiftool-vendored**](https://github.com/photostructure/exiftool-vendored.js "null") for comprehensive tag reading.
    
-   **Local Machine Learning Embeddings:** Generates semantic image vectors locally using ONNX runtime and [**Transformers.js**](https://huggingface.co/docs/transformers.js "null") (`Xenova/clip-vit-base-patch32`), ideal for building semantic image search or recommendation engines.
    
-   **UI Enhancements:** Generates ultra-fast low-res base64 blur placeholders and dominant color palettes using [**sharp**](https://sharp.pixelplumbing.com/ "null").
    
-   **Cloud Upload Sync:** Offloads asset distribution to remote CDNs during production builds while maintaining local URLs during development.
    
-   **Race-Condition Safe:** Implements a promise-locking mechanism to guarantee thread/hook-safe evaluation inside parallel Vite or Astro build environments.
    

## Installation

Install the package via your preferred package manager:

```bash
npm install vite-image-pipeline
```

## Setup & Integration

### For Astro Projects

Add the helper integration to your `astro.config.mjs` file to ensure the backend processes safely shut down and remote asset synchronization executes automatically during `astro:build:done`.

```javascript
import { defineConfig } from 'astro/config';
import { astroImagePipelinePlugin } from 'vite-image-pipeline';

export default defineConfig({
  integrations: [
    astroImagePipelinePlugin()
  ]
});
```

### For Non-Astro Vite Projects (React, Vue, Svelte, etc.)

Because the pipeline uses a robust programmatic API, you can import and run the utility functions anywhere in your build scripts, dev servers, or server-side frameworks ([**SvelteKit**](https://kit.svelte.dev/ "null"), [**Next.js**](https://nextjs.org/ "null"), [**Nuxt**](https://nuxt.com/ "null"), etc.):

```javascript
import { getEmbeddings, getImageColors } from 'vite-image-pipeline';

// Use directly in your data-loading scripts, API endpoints, or pre-render hooks
```

### Configure Options (Optional)

You can customize cache locations, batch processing sizes, and the target machine learning model globally at the entry point of your pipeline execution:

```javascript
import { setOptions } from 'vite-image-pipeline';

setOptions({
    modelName: "Xenova/clip-vit-base-patch32", // Target embedding model
    batchSize: 4,                              // Concurrent images sent to ML inference
    metadataCachePath: "./.custom-cache/metadata.json",
    embeddingCachePath: "./.custom-cache/embeddings.json"
});
```

## API Reference & Usage Examples

### Metadata Extraction

Reads embedded EXIF/IPTC metadata information out of your original source files.
```javascript
import { getMetadata } from 'vite-image-pipeline';

const images = ['src/assets/photo1.jpg', 'src/assets/photo2.jpg'];
const metadataMap = await getMetadata(images);

const photo1Tags = metadataMap['src/assets/photo1.jpg'];
console.log(`Shot taken with camera: ${photo1Tags.Model}`);
```

### Machine Learning Embeddings

Generates normalized vector embeddings of your images. These arrays can be passed straight into vector databases (like [**Chroma**](https://www.trychroma.com/ "null"), [**Pinecone**](https://www.pinecone.io/ "null"), or [**pgvector**](https://github.com/pgvector/pgvector "null")) to enable image-to-image or text-to-image searching.

    import { getEmbeddings } from 'vite-image-pipeline';
    
    const embeddingsMap = await getEmbeddings(['src/assets/photo1.jpg']);
    const vector = embeddingsMap['src/assets/photo1.jpg']; 
    // Output: Array of numbers (e.g., length 512 for CLIP)
    

### Low-Resolution Blur Placeholders

Generates a small, base64 encoded Data URI placeholder to minimize [**Cumulative Layout Shift (CLS)**](https://web.dev/cls/ "null") and enable sleek "blur-up" loading configurations.

```javascript
import { getImageBlurPlaceholders } from 'vite-image-pipeline';

const blurMap = await getImageBlurPlaceholders(['src/assets/photo1.jpg']);
const placeholderUri = blurMap['src/assets/photo1.jpg'];
// Output: "data:image/jpeg;base64,/9j/4AAQSkZJR..."
```
    

### Dominant Color Sampling

Extracts the mathematical dominant color from an image. This is highly useful for assigning container background fallbacks before images load, or tinting modern UI wrappers dynamically.

```javascript
import { getImageColors } from 'vite-image-pipeline';
    
const colorsMap = await getImageColors(['src/assets/photo1.jpg']);
const { r, g, b } = colorsMap['src/assets/photo1.jpg'];
console.log(`Dominant RGB: rgb(${r}, ${g}, ${b})`);
```

### Remote Cloud Upload Registry

Queues files for production CDN synchronization. By default, this mechanism gracefully falls back to returning local paths during development (`import.meta.env.PROD === false`) and fires off multi-platform uploads sequentially when triggered (or automatically during the Astro build lifecycle).

```javascript
import { uploadRemoteImages } from 'vite-image-pipeline';

const remoteOptions = {
    platform: 's3', // configuration schema depends on your remote implementation
    bucket: 'my-production-cdn'
};

const transformedUrls = await uploadRemoteImages(
    remoteOptions, 
    ['src/assets/photo1.jpg']
);
// In Dev:  ['src/assets/photo1.jpg']
// In Prod: ['[https://my-production-cdn.s3.amazonaws.com/assets/photo1.jpg](https://my-production-cdn.s3.amazonaws.com/assets/photo1.jpg)']
```
    

## Architectural Compatibility

This pipeline operates on the **source assets** and extracts structural insights or registers cloud delivery rules. It does not replace UI optimization tooling.

### Synergy with `vite-imagetools` or Astro `<Image />`

You should use this pipeline alongside your visual optimization workflows to achieve high performance combined with smart data features:

#### Example in Astro

```astro
---
// src/pages/gallery.astro
import { Image } from 'astro:assets';
import { getImageBlurPlaceholders, getImageColors } from 'vite-image-pipeline';
import localImage from '../assets/hero.jpg';

// 1. Run pipeline tasks using absolute path references
const assetPath = localImage.fsPath; 
const blurMap = await getImageBlurPlaceholders([assetPath]);
const colorMap = await getImageColors([assetPath]);

const placeholder = blurMap[assetPath];
const dominantColor = colorMap[assetPath];
---

<div 
    class="image-container" 
    style={`background-color: rgb(${dominantColor.r}, ${dominantColor.g}, ${dominantColor.b});`}
>
    <!-- Use Astro's standard image component for final HTML generation & resizing -->
    <Image 
    src={localImage} 
    alt="Hero visual description" 
    style={`background-image: url(${placeholder}); background-size: cover;`} 
    />
</div>
```

#### Example in React + `vite-imagetools`

```javascript
// src/components/Hero.jsx
import React from 'react';
// Import resized/optimized images with vite-imagetools query params
import optimizedHero from '../assets/hero.jpg?width=800&format=webp';

export function Hero({ blurPlaceholder, dominantColor }) {
    return (
    <div 
        className="hero-wrapper" 
        style={{ 
        backgroundColor: `rgb(${dominantColor.r}, ${dominantColor.g}, ${dominantColor.b})` 
        }}
    >
        <img 
        src={optimizedHero} 
        alt="Hero image"
        style={{
            backgroundImage: `url(${blurPlaceholder})`,
            backgroundSize: 'cover',
        }}
        />
    </div>
    );
}
```
# vite-image-pipeline

A high-performance, asset-caching image pipeline designed for **[Vite](https://vite.dev/)**-based projects (**[Astro](https://astro.build/)**, **[React](https://react.dev/)**, **[Svelte](https://svelte.dev/)**, **[Vue](https://vuejs.org/)**, **[Solid](https://www.solidjs.com/)**, and more). It extracts rich metadata, generates local ML-driven vector embeddings, computes low-resolution blur placeholders, samples dominant colors, and handles zero-egress remote platform uploads during production builds.

While `vite-image-pipeline` is completely framework-agnostic and can be utilized in any Vite context, it also includes a dedicated integration for **[Astro](https://astro.build/)** to automatically coordinate cleanup and remote synchronization on production builds.

## Why Use This?

This package is designed to work seamlessly **in conjunction with** image compression and optimization tools such as **[vite-imagetools](https://github.com/JonasKruckenberg/imagetools)** or Astro's native **[Image Component](https://docs.astro.build/en/guides/images/)**.

Instead of replacing image optimizers, it splits the pipeline into two parallel, specialized paths that merge gracefully at the UI layer:

```
                      [ Source Image Asset ]
                                │
            ┌───────────────────┴───────────────────┐
            ▼                                       ▼
┌───────────────────────┐               ┌───────────────────────┐
│  vite-image-pipeline  │               │   vite-imagetools /   │
│  (Data Extraction)    │               │    Astro's <Image>    │
├───────────────────────┤               ├───────────────────────┤
│ • EXIF Metadata       │               │ • Multi-format output │
│ • Blur Placeholders   │               │   (WebP, AVIF, JPEG)  │
│ • Dominant Colors     │               │ • Fluid UI Resizing   │
│ • Semantic Vectors    │               │ • File Compression    │
└───────────┬───────────┘               └───────────┬───────────┘
            │                                       │
            ▼                                       ▼
 [ JSON Cache / Cloud CDN ]             [ Optimized Web Assets ]
            │                                       │
            └───────────────────┬───────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │      Frontend UI      │
                    │ (Astro / React / etc) │
                    ├───────────────────────┤
                    │ • Low-CLS Blur loading│
                    │ • Dynamic BG Tints    │
                    │ • Semantic Search     │
                    └───────────────────────┘

```

It processes your original source assets to power rich frontend features like **semantic visual search**, **dynamic UI color-matching**, and **custom lazy-loading states**, leaving final layout rendering, sizing, and responsive resizing to your preferred UI layer.

## Features

* **Flexible Path Resolution:** Fully supports arbitrary filesystem structures. Paths passed to data extraction functions can be project-relative (e.g., `/src/assets/photo.jpg`), project-root relative (e.g., `src/assets/photo.jpg`), or absolute global paths (e.g., `/Users/username/Desktop/photo.jpg`).
* **Aggressive Content-Based Caching:** Cross-references file modification times (`mtime`), sizes, and fast initial-block cryptographic hashes to guarantee assets are processed exactly once unless edited.
* **EXIF Metadata Extraction:** Powered by **[exiftool-vendored](https://github.com/photostructure/exiftool-vendored.js)** for lightning-fast, comprehensive tag reading (Camera model, lens, exposure, timestamps, geo-coordinates).
* **Local Machine Learning Embeddings:** Generates normalized semantic image vectors locally using ONNX runtime and **[Transformers.js](https://huggingface.co/docs/transformers.js)** (`Xenova/clip-vit-base-patch32`). Perfect for image-to-image or text-to-image similarity matching.
* **UI Enhancements:** Generates ultra-fast low-res base64 blur placeholders and dominant color palettes using native `sharp` bindings.
* **Cloud Upload Sync & Zero-Egress Caching:** Offloads assets to remote S3-compatible endpoints (like Cloudflare R2) sequentially during production builds. Leverages `IfNoneMatch` ETag validation to skip uploading unmodified files and purges local build items post-upload to keep your bundle footprint tiny.
* **Race-Condition Safe:** Implements an internal promise-locking sequence to guarantee thread/hook-safe evaluation inside heavily parallelized Vite or Astro multi-threaded builds.

## Installation

Install the package via your preferred package manager:

```bash
npm install vite-image-pipeline

```

## Path Resolution Rules

The image pipeline intelligently handles the formatting differences introduced by bundler hooks and components across frameworks:

1. **Project-Relative Paths (`/src/*`):** Resolves paths starting with `/src/` straight against your current working directory (`process.cwd()`).
2. **Absolute Global Paths:** Fully supports native system roots (`/Users/`, `/home/`, or Windows drive letters) for scenarios where images are processed out of external directories.
3. **Vite Internals (`/@fs/*`):** Automatically cleans up internal prefixes injected by Vite's dev server to match actual file boundaries cleanly.

```javascript
import { getMetadata } from 'vite-image-pipeline';

// All map keys are fully compatible and safely mapped:
const data = await getMetadata([
  '/src/assets/gallery/photo1.jpg',                // Project relative
  'src/assets/gallery/photo2.jpg',                 // Root relative
  '/Users/alex/Projects/site/src/assets/photo3.jpg' // Absolute global path
]);

```

## Advanced Configuration

You can customize cache paths, concurrency profiles, and machine learning runtime characteristics globally at the entry point of your pipeline execution using `setOptions`.

### Complete Options Schema

```javascript
import { setOptions } from 'vite-image-pipeline';

setOptions({
  // The HuggingFace/Transformers.js repository string for feature extraction
  modelName: "Xenova/clip-vit-base-patch32", 
  
  // Number of images processed concurrently during ML vector inference
  batchSize: 4,                              
  
  // Fully qualified filesystem targets for the localized JSON content-caches
  metadataCachePath: "./.image-pipeline/metadata-cache.json",
  embeddingCachePath: "./.image-pipeline/embedding-cache.json",
  colorCachePath: "./.image-pipeline/color-cache.json",
  blurCachePath: "./.image-pipeline/blur-cache.json",
  
  // Target directory where local ONNX weights and tokenizer models are written
  modelCachePath: "./.image-pipeline/models"
});

```

## Full API Reference & Examples

### 1. Metadata Extraction

Reads embedded EXIF, IPTC, and XMP metadata out of original source files.

```javascript
import { getMetadata } from 'vite-image-pipeline';

const images = ['src/assets/photo1.jpg', 'src/assets/photo2.jpg'];
const metadataMap = await getMetadata(images);

const photo1 = metadataMap['src/assets/photo1.jpg'];
console.log(`Camera: ${photo1.Model}, Lens: ${photo1.LensModel}, ISO: ${photo1.ISO}`);

```

### 2. Machine Learning Embeddings

Generates a normalized 512-dimensional vector embedding array. These arrays can be sent straight into vector stores (like Chroma, Pinecone, or pgvector) for semantic searching.

```javascript
import { getEmbeddings } from 'vite-image-pipeline';

const embeddingsMap = await getEmbeddings(['src/assets/photo1.jpg']);
const vector = embeddingsMap['src/assets/photo1.jpg']; 
// Output: [0.0124, -0.0452, 0.0911, ... 512 floats long]

```

### 3. Low-Resolution Blur Placeholders

Generates a `20px` wide blurred base64 JPEG/PNG Data URI to avoid Cumulative Layout Shift (CLS) during image lazy-loading.

```javascript
import { getImageBlurPlaceholders } from 'vite-image-pipeline';

const blurMap = await getImageBlurPlaceholders(['src/assets/photo1.jpg']);
const placeholderUri = blurMap['src/assets/photo1.jpg'];
// Output: "data:image/jpeg;base64,/9j/4AAQSkZJR..."

```

### 4. Dominant Color Sampling

Extracts the principal structural RGB values of an image to match container background colors before asset load execution completes.

```javascript
import { getImageColors } from 'vite-image-pipeline';

const colorsMap = await getImageColors(['src/assets/photo1.jpg']);
const { r, g, b } = colorsMap['src/assets/photo1.jpg'];
console.log(`Dominant Background: rgb(${r}, ${g}, ${b})`);

```

### 5. Remote Cloud Upload Registry

Queues assets to pass onto remote storage during production compiles. Automatically bypasses uploads during development mode (`import.meta.env.PROD === false`).

```javascript
import { uploadRemoteImages } from 'vite-image-pipeline';

const remoteOptions = {
  platform: 'cloudflare-r2',
  accountId: process.env.R2_ACCOUNT_ID,
  r2AccessKey: process.env.R2_ACCESS_KEY,
  r2SecretKey: process.env.R2_SECRET_KEY,
  bucketName: 'my-gallery-cdn',
  outDir: 'dist' // Local output distribution path to read from and prune
};

const transformedUrls = await uploadRemoteImages(remoteOptions, ['src/assets/photo1.jpg']);
// Dev output:  ['src/assets/photo1.jpg']
// Prod output: ['https://my-gallery-cdn.r2.cloudflarestorage.com/src/assets/photo1.jpg']

```

## Comprehensive Implementation Recipes

### Astro (Production SSG / SSR)

Add the integration manager hook to your `astro.config.mjs` setup to coordinate graceful engine shutdown (`exiftool` worker pools) and handle file uploading sequences during `astro:build:done`.

```javascript
// astro.config.mjs
import { defineConfig } from 'astro/config';
import { astroImagePipelinePlugin } from 'vite-image-pipeline';

export default defineConfig({
  integrations: [
    astroImagePipelinePlugin()
  ]
});

```

#### Inside an Astro Component (`src/components/SmartImage.astro`)

```astro
---
import { Image } from 'astro:assets';
import { getImageBlurPlaceholders, getImageColors, getMetadata } from 'vite-image-pipeline';

interface Props {
  imageAsset: any; // Astro ESM Image Import Reference
  alt: string;
}

const { imageAsset, alt } = Astro.props;

// 1. Resolve structural filesystem pathing safely
const absolutePath = imageAsset.fsPath;

// 2. Fetch data parallelized via content caching locks
const [blurMap, colorMap, metaMap] = await Promise.all([
  getImageBlurPlaceholders([absolutePath]),
  getImageColors([absolutePath]),
  getMetadata([absolutePath])
]);

const blurPlaceholder = blurMap[absolutePath];
const { r, g, b } = colorMap[absolutePath];
const cameraModel = metaMap[absolutePath]?.Model || "Unknown Camera";
---

<div 
  class="image-wrapper" 
  style={`background-color: rgb(${r}, ${g}, ${b}); position: relative; overflow: hidden;`}
>
  <Image 
    src={imageAsset} 
    alt={alt}
    loading="lazy"
    style={`background-image: url(${blurPlaceholder}); background-size: cover;`}
  />
  <span class="exif-overlay">Shot on: {cameraModel}</span>
</div>

<style>
  .image-wrapper img { transition: filter 0.3s ease-out; }
  .exif-overlay { position: absolute; bottom: 8px; left: 8px; color: #fff; font-size: 0.75rem; }
</style>

```

### React + Vite (With `vite-imagetools`)

In non-Astro build architectures, call the data functions directly within your build runner, build plugins, data-loading environments, or node generation engines.

#### Step 1: Pre-render Data Config Script (`scripts/process-images.js`)

Run this process script before or during your main build loop to extract assets details into localized JSON references:

```javascript
import fs from 'fs/promises';
import glob from 'fast-glob';
import { getImageBlurPlaceholders, getImageColors } from 'vite-image-pipeline';

async function buildStaticManifest() {
  const filePaths = await glob('src/assets/gallery/*.{jpg,jpeg,png,webp,avif}');
  
  const [blurs, colors] = await Promise.all([
    getImageBlurPlaceholders(filePaths),
    getImageColors(filePaths)
  ]);

  const manifest = filePaths.reduce((acc, rawPath) => {
    acc[rawPath] = {
      blur: blurs[rawPath],
      color: colors[rawPath]
    };
    return acc;
  }, {});

  await fs.writeFile('./src/image-manifest.json', JSON.stringify(manifest, null, 2));
  console.log('⚡ Image pipeline data manifest compiled.');
}

buildStaticManifest();

```

#### Step 2: Consume Data inside your React UI Engine

```jsx
import React from 'react';
// 1. Load optimized images via query-params from vite-imagetools
import optimizedHero from '../assets/gallery/lake-sunset.jpg?width=1200&format=avif';
// 2. Import the statically generated metadata mapping file
import imageManifest from '../image-manifest.json';

const ASSET_KEY = 'src/assets/gallery/lake-sunset.jpg';

export function HighPerformanceHero() {
  const uiMeta = imageManifest[ASSET_KEY] || { blur: '', color: { r: 30, g: 30, b: 30 } };
  const { r, g, b } = uiMeta.color;

  return (
    <div 
      className="hero-container" 
      style={{ backgroundColor: `rgb(${r}, ${g}, ${b})`, minHeight: '400px' }}
    >
      <img 
        src={optimizedHero} 
        alt="Lake Sunset" 
        loading="lazy"
        style={{
          backgroundImage: `url(${uiMeta.blur})`,
          backgroundSize: 'cover',
          width: '100%',
          height: 'auto'
        }}
      />
    </div>
  );
}

```

## Troubleshooting

### Machine Learning Performance Tweaks

When compiling in standard local environments, `Transformers.js` splits processes dynamically over multicore nodes. If processing locks up execution inside CI/CD cloud actions (like GitHub Actions runners or Vercel build instances), pin execution to a single worker profile core manually:

```bash
NODE_ENV=production npm run build

```

*(The engine sets `wasm.numThreads = 1` dynamically when detecting `production` variables to avoid thread overhead allocation crashes).*

### Cloudflare R2 Upload Checksums 412 Errors

If you see skipped execution responses reporting `PreconditionFailed` or `412` HTTP codes during automated deployment pipelines, **this behavior is intentional**. It guarantees that files whose cryptographic hashes match assets currently hosted on your R2 object storage CDN avoid re-upload workflows, saving you bandwidth and billing resource allocations.
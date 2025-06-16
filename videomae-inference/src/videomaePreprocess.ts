// Utility functions for VideoMAE preprocessing

// ImageNet mean and std
export const IMAGENET_MEAN = [0.485, 0.456, 0.406];
export const IMAGENET_STD = [0.229, 0.224, 0.225];

// Resize and normalize a canvas image to 224x224 and return Float32Array [3, 224, 224]
export function preprocessFrame(canvas: HTMLCanvasElement): Float32Array {
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('No 2D context');
  const imageData = ctx.getImageData(0, 0, 224, 224).data;
  const floatArray = new Float32Array(3 * 224 * 224);
  for (let y = 0; y < 224; y++) {
    for (let x = 0; x < 224; x++) {
      const idx = (y * 224 + x) * 4;
      // RGB order
      for (let c = 0; c < 3; c++) {
        let value = imageData[idx + c] / 255;
        value = (value - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
        floatArray[c * 224 * 224 + y * 224 + x] = value;
      }
    }
  }
  return floatArray;
}

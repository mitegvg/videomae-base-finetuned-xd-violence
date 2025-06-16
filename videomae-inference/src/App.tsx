import { useState, useRef } from 'react';
import * as ort from 'onnxruntime-web';
import './App.css';
import { loadOnnxModel, runVideoMAEInference } from './videomaeOnnx';
// Import IMAGENET_MEAN and IMAGENET_STD as well
import { preprocessFrame, IMAGENET_MEAN, IMAGENET_STD } from './videomaePreprocess';

// Add these constants
const CLASS_LABELS = ['A', 'B1', 'B1-0-0', 'B1-B2-0', 'B1-B2-B5', 'B1-B2-B6', 'B1-B2-G', 'B1-B4-0', 'B1-B5-0', 'B1-B5-B6', 'B1-B6-0', 'B1-G-0', 'B2-0-0', 'B2-B1-0', 'B2-B1-B5', 'B2-B5-0', 'B2-B5-G', 'B2-B6-0', 'B2-B6-B1', 'B2-B6-G', 'B2-G-0', 'B2-G-B1', 'B2-G-B6', 'B4-0-0', 'B4-B1-0', 'B4-B1-G', 'B4-B2-0', 'B4-B2-B1', 'B4-B5-B1', 'B5-0-0', 'B5-B1-0', 'B5-B1-B2', 'B5-B2-0', 'B6-0-0', 'B6-B2-0', 'B6-B2-G', 'B6-B4-0', 'B6-G-0', 'B6-G-B2', 'G', 'G-0-0', 'G-B1-0', 'G-B2-0', 'G-B2-B1', 'G-B2-B6', 'G-B6-0'];
const LABEL_EXPLANATIONS: Record<string, string> = {
  'A': 'Nothing detected',
  'B1': 'Fighting',
  'B2': 'Shooting',
  'B4': 'Riot',
  'B5': 'Abuse',
  'B6': 'Car accident',
  'G': 'Explosion'
};

// Helper function to denormalize and save a frame as an image
function saveFrameAsImage(
  normalizedFrameData: Float32Array,
  frameIndex: number,
  mean: number[],
  std: number[]
) {
  const width = 224;
  const height = 224;
  const denormalizedData = new Uint8ClampedArray(width * height * 4);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      // Accessing data in C,H,W format: data[channel * height * width + row * width + col]
      const R_norm = normalizedFrameData[0 * height * width + y * width + x];
      const G_norm = normalizedFrameData[1 * height * width + y * width + x];
      const B_norm = normalizedFrameData[2 * height * width + y * width + x];

      // Denormalize and clamp values to [0, 255]
      const R_denorm = Math.min(255, Math.max(0, Math.round((R_norm * std[0] + mean[0]) * 255)));
      const G_denorm = Math.min(255, Math.max(0, Math.round((G_norm * std[1] + mean[1]) * 255)));
      const B_denorm = Math.min(255, Math.max(0, Math.round((B_norm * std[2] + mean[2]) * 255)));

      const idx = (y * width + x) * 4;
      denormalizedData[idx + 0] = R_denorm;
      denormalizedData[idx + 1] = G_denorm;
      denormalizedData[idx + 2] = B_denorm;
      denormalizedData[idx + 3] = 255; // Alpha channel (fully opaque)
    }
  }

  const imageData = new ImageData(denormalizedData, width, height);
  const tempSaveCanvas = document.createElement('canvas');
  tempSaveCanvas.width = width;
  tempSaveCanvas.height = height;
  const tempSaveCtx = tempSaveCanvas.getContext('2d');

  if (tempSaveCtx) {
    tempSaveCtx.putImageData(imageData, 0, 0);
    const link = document.createElement('a');
    link.href = tempSaveCanvas.toDataURL('image/png');
    link.download = `processed_frame_${frameIndex}.png`;
    document.body.appendChild(link); // Append to body to ensure click works in all browsers (esp. Firefox)
    link.click();
    document.body.removeChild(link); // Clean up the link
  }
}

function App() {
  const [model, setModel] = useState<ort.InferenceSession | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);
  // Correctly access the web version for ONNX Runtime Web
  const ortVersion = ort.env.versions.web; 

  // Set WASM paths to CDN
  // Ensure ortVersion is defined before using it in the template string
  if (ortVersion) {
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ortVersion}/dist/`;
    console.log('ONNX Runtime Web WASM paths set to:', ort.env.wasm.wasmPaths);
  } else {
    // Fallback or default path if ortVersion is somehow undefined
    // This is a safeguard, typically ort.env.versions.web should be defined.
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/'; 
    console.warn('ONNX Runtime Web version undefined, using default WASM path.', ort.env.wasm.wasmPaths);
  }

  const modelUrl = './videomae.onnx'; 
  const videoSrc = './test1B1-0-0.mp4'; 

  const handleInfer = async () => {
    if (!videoRef.current) {
      setLog(l => [...l, 'Video not loaded']);
      return;
    }
    setLog(['Starting inference...']);
    let currentSession = model;

    try {
      if (!model) { 
        setLog(l => [...l, 'Loading ONNX model...']);
        const loadedSession = await loadOnnxModel(modelUrl);
        if (loadedSession) {
          setModel(loadedSession);
          setLog(l => [...l, `Model loaded. Input names: ${loadedSession.inputNames.join(', ')}`]);
          console.log(`Model loaded. Input names: ${loadedSession.inputNames.join(', ')}`);
          console.log('Expected input shape by model (from prior errors/docs): [1, 16, 3, 224, 224] for input', loadedSession.inputNames[0]);
          currentSession = loadedSession; 
        } else {
          setLog(l => [...l, 'Failed to load the ONNX model.']);
          console.error('Failed to load the ONNX model.');
          return;
        }
      } else {
        currentSession = model; 
      }

      const video = videoRef.current;
      video.pause();
      video.currentTime = 0;
      setLog(l => [...l, 'Video paused and reset to start.']);

      const canvas = document.createElement('canvas');
      canvas.width = 224;
      canvas.height = 224;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) {
        setLog(l => [...l, 'Failed to get canvas context']);
        return;
      }

      const numFramesToExtract = 16;
      const processedFrames: Float32Array[] = [];
      const duration = video.duration;
      setLog(l => [...l, `Video duration: ${duration.toFixed(2)}s. Extracting ${numFramesToExtract} frames.`]);

      // Create a temporary canvas for resizing and cropping operations
      const tempCanvas = document.createElement('canvas');
      const tempCtx = tempCanvas.getContext('2d');
      if (!tempCtx) {
        setLog(l => [...l, 'Failed to get temporary canvas context']);
        return;
      }

      for (let i = 0; i < numFramesToExtract; i++) {
        const time = (i / (numFramesToExtract -1)) * duration;
        video.currentTime = time;
        await new Promise(resolve => { video.onseeked = resolve; });

        // Original video dimensions
        const vW = video.videoWidth;
        const vH = video.videoHeight;

        // Calculate new dimensions for resizing shortest edge to 224, maintaining aspect ratio
        let resizeWidth, resizeHeight;
        if (vW < vH) { // Width is the shortest edge
            const scale = 224 / vW;
            resizeWidth = 224;
            resizeHeight = Math.round(vH * scale);
        } else { // Height is the shortest edge, or they are equal
            const scale = 224 / vH;
            resizeHeight = 224;
            resizeWidth = Math.round(vW * scale);
        }

        // Set temporary canvas dimensions and draw the resized video frame
        tempCanvas.width = resizeWidth;
        tempCanvas.height = resizeHeight;
        tempCtx.drawImage(video, 0, 0, resizeWidth, resizeHeight);

        // Calculate cropping coordinates for center crop (from the resized image)
        const cropX = Math.max(0, (resizeWidth - 224) / 2);
        const cropY = Math.max(0, (resizeHeight - 224) / 2);

        // Clear the main 224x224 canvas (ctx) before drawing the new cropped image
        ctx.clearRect(0, 0, 224, 224);
        // Draw the center-cropped image from tempCanvas onto the main 224x224 canvas (ctx)
        // The source rectangle is (cropX, cropY, 224, 224) from tempCanvas
        // The destination rectangle is (0, 0, 224, 224) on the main canvas
        ctx.drawImage(tempCanvas, cropX, cropY, 224, 224, 0, 0, 224, 224);
        
        // Pass the main 224x224 canvas (which now holds the correctly resized and cropped frame)
        const frameData = preprocessFrame(canvas); 
        processedFrames.push(frameData);
        // Save the denormalized frame for inspection
        saveFrameAsImage(frameData, i, IMAGENET_MEAN, IMAGENET_STD);
        setLog(l => [...l, `Processed frame ${i + 1}/${numFramesToExtract} at time ${time.toFixed(2)}s. Saved.`]);
      }
      setLog(l => [...l, 'All frames processed.']);

      const inputTensorData = new Float32Array(numFramesToExtract * 3 * 224 * 224);
      let offset = 0;
      for (const frame of processedFrames) {
        inputTensorData.set(frame, offset);
        offset += frame.length;
      }
      setLog(l => [...l, `Input tensor data prepared. Shape: [${numFramesToExtract}, 3, 224, 224], total elements: ${inputTensorData.length}`]);

      if (!currentSession) {
        setLog(l => [...l, 'ONNX Session not available.']);
        console.error('ONNX Session not available.');
        return;
      }

      setLog(l => [...l, 'Running inference...']);
      const outputTensor = await runVideoMAEInference(currentSession, inputTensorData);
      setLog(l => [...l, 'Inference complete.']);
      console.log('Output Tensor:', outputTensor);

      const outputArray = Array.from(outputTensor);
      const probabilities = outputArray.map((logit, index) => ({ logit, index }));
      probabilities.sort((a, b) => b.logit - a.logit); 

      let resultsLog = ['Top-5 predictions:'];
      for (let i = 0; i < Math.min(5, probabilities.length); i++) {
        const classIndex = probabilities[i].index;
        const score = probabilities[i].logit;
        const className = CLASS_LABELS[classIndex] || 'Unknown';
        
        let explanation = '';
        if (className !== 'A' && className !== 'Unknown') {
          const subLabels = className.split('-');
          const explanations = subLabels.map(sl => LABEL_EXPLANATIONS[sl] || sl);
          explanation = explanations.join(', ');
        } else {
          explanation = LABEL_EXPLANATIONS[className] || 'No specific event';
        }
        resultsLog.push(`Class: ${className} (Index: ${classIndex}), Score: ${score.toFixed(4)}, Description: ${explanation}`);
      }
      setLog(l => [...l, ...resultsLog]);

    } catch (error) {
      console.error('Error during inference:', error);
      setLog(l => [...l, `Error: ${(error as Error).message}`]);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>VideoMAE ONNX Inference</h1>
        <video ref={videoRef} width="320" height="240" controls src={videoSrc}></video>
        <button onClick={handleInfer} style={{ marginTop: '10px', padding: '10px' }}>Run Inference</button>
        <div style={{ marginTop: '20px', textAlign: 'left', fontFamily: 'monospace', whiteSpace: 'pre-wrap', maxHeight: '400px', overflowY: 'auto', border: '1px solid #ccc', padding: '10px' }}>
          {log.join('\n')}
        </div>
      </header>
    </div>
  );
}

export default App;

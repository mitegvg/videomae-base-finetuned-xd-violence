import * as ort from 'onnxruntime-web';

export async function loadOnnxModel(modelUrl: string) {
  return await ort.InferenceSession.create(modelUrl);
}

export async function runVideoMAEInference(
  session: ort.InferenceSession,
  inputTensor: Float32Array // Flat data from App.tsx, ordered as T,C,H,W (16,3,224,224)
) {
  // The ONNX model expects input shape [Batch, Frames, Channels, Height, Width]
  // Based on the error: "index: 1 Got: 3 Expected: 16", "index: 2 Got: 16 Expected: 3"
  // The model expects [1, 16, 3, 224, 224]
  const modelExpectedShape = [1, 16, 3, 224, 224];
  const tensor = new ort.Tensor('float32', inputTensor, modelExpectedShape);

  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = tensor;
  const results = await session.run(feeds);
  // Assume output is the first output
  const output = results[session.outputNames[0]];
  return output.data as Float32Array;
}

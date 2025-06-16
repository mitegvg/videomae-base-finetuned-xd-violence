\
import cv2
import numpy as np
import onnxruntime

# Constants from the frontend
CLASS_LABELS = ['A', 'B1', 'B1-0-0', 'B1-B2-0', 'B1-B2-B5', 'B1-B2-B6', 'B1-B2-G', 'B1-B4-0', 'B1-B5-0', 'B1-B5-B6', 'B1-B6-0', 'B1-G-0', 'B2-0-0', 'B2-B1-0', 'B2-B1-B5', 'B2-B5-0', 'B2-B5-G', 'B2-B6-0', 'B2-B6-B1', 'B2-B6-G', 'B2-G-0', 'B2-G-B1', 'B2-G-B6', 'B4-0-0', 'B4-B1-0', 'B4-B1-G', 'B4-B2-0', 'B4-B2-B1', 'B4-B5-B1', 'B5-0-0', 'B5-B1-0', 'B5-B1-B2', 'B5-B2-0', 'B6-0-0', 'B6-B2-0', 'B6-B2-G', 'B6-B4-0', 'B6-G-0', 'B6-G-B2', 'G', 'G-0-0', 'G-B1-0', 'G-B2-0', 'G-B2-B1', 'G-B2-B6', 'G-B6-0']
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NUM_FRAMES_TO_EXTRACT = 16
TARGET_SIZE = 224

def preprocess_frame_python(frame_rgb):
    # Resize shortest edge to TARGET_SIZE
    h, w, _ = frame_rgb.shape
    if h < w:
        new_h = TARGET_SIZE
        new_w = int(w * (TARGET_SIZE / h))
    else:
        new_w = TARGET_SIZE
        new_h = int(h * (TARGET_SIZE / w))
    
    resized_frame = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center crop
    y_offset = (new_h - TARGET_SIZE) // 2
    x_offset = (new_w - TARGET_SIZE) // 2
    cropped_frame = resized_frame[y_offset:y_offset + TARGET_SIZE, x_offset:x_offset + TARGET_SIZE]

    # Normalize
    # Convert to float and scale to 0-1
    normalized_frame = cropped_frame.astype(np.float32) / 255.0
    normalized_frame = (normalized_frame - IMAGENET_MEAN) / IMAGENET_STD
    
    # Transpose from HWC to CHW
    chw_frame = normalized_frame.transpose(2, 0, 1)
    return chw_frame

def main():
    video_path = 'public/test0.mp4'
    onnx_model_path = 'public/videomae.onnx'

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_ms = (total_frames / fps) * 1000
    
    print(f"Video: {video_path}, Duration: {duration_ms/1000:.2f}s, FPS: {fps:.2f}, Total Frames: {total_frames}")

    processed_frames_list = []

    for i in range(NUM_FRAMES_TO_EXTRACT):
        time_ms = (i / (NUM_FRAMES_TO_EXTRACT - 1)) * duration_ms if NUM_FRAMES_TO_EXTRACT > 1 else 0
        # For the last frame, ensure we don't go past the video duration
        if i == NUM_FRAMES_TO_EXTRACT - 1:
            time_ms = min(time_ms, duration_ms - (1000/fps)) # Go to just before the end

        cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame at time {time_ms:.2f}ms (frame index {i}). Using last good frame or breaking.")
            if not processed_frames_list: # No frames processed yet
                 print("Error: Failed to read any frames.")
                 cap.release()
                 return
            # If some frames were already processed, use the last good one.
            # This might not be ideal but prevents crashing if video ends slightly early.
            processed_frames_list.append(processed_frames_list[-1].copy())
            continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        chw_frame = preprocess_frame_python(frame_rgb)
        processed_frames_list.append(chw_frame)
        print(f"Processed frame {i+1}/{NUM_FRAMES_TO_EXTRACT} at time {time_ms/1000:.2f}s")

    cap.release()

    if len(processed_frames_list) != NUM_FRAMES_TO_EXTRACT:
        print(f"Error: Expected {NUM_FRAMES_TO_EXTRACT} frames, but got {len(processed_frames_list)}")
        return

    # Stack frames and add batch dimension: (1, T, C, H, W)
    input_tensor_data = np.stack(processed_frames_list, axis=0)
    input_tensor_data = np.expand_dims(input_tensor_data, axis=0).astype(np.float32)
    print(f"Input tensor shape: {input_tensor_data.shape}")

    # Run ONNX inference
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = ort_session.get_inputs()[0].name
    
    print(f"Running inference with ONNX model: {onnx_model_path}")
    results = ort_session.run(None, {input_name: input_tensor_data})
    output_logits = results[0][0] # Assuming batch size 1, get the logits for the first (only) item
    print(f"Output: {results}")
    # Get top-5 predictions
    # Softmax to get probabilities (optional, for understanding, logits are usually used for argmax)
    # exp_logits = np.exp(output_logits - np.max(output_logits)) 
    # probabilities = exp_logits / np.sum(exp_logits)
    
    # Get top 5 logit values and their indices
    top_k_indices = np.argsort(output_logits)[::-1][:5]

    print("\\nTop-5 predictions (Python ONNX):")
    for idx in top_k_indices:
        class_name = CLASS_LABELS[idx] if idx < len(CLASS_LABELS) else "Unknown"
        logit_score = output_logits[idx]
        print(f"Class: {class_name} (Index: {idx}), Logit Score: {logit_score:.4f}")

if __name__ == '__main__':
    main()

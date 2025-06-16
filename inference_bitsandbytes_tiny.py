import os
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor, BitsAndBytesConfig
import glob
import av
import numpy as np

def read_video_pyav(container, num_frames_to_sample: int):
    """Decodes video frames from a PyAV container."""
    frames = []
    # Ensure the video stream has enough frames
    if container.streams.video[0].frames == 0: # Some videos might not report frames correctly
        # Fallback: try to decode up to num_frames_to_sample * some_factor frames
        # This is a heuristic for videos with missing frame count metadata
        max_frames_to_try_decode = num_frames_to_sample * 5 
        decoded_count = 0
        for i, frame in enumerate(container.decode(video=0)):
            if decoded_count < num_frames_to_sample:
                frames.append(frame.to_ndarray(format="rgb24"))
                decoded_count +=1
            if i >= max_frames_to_try_decode -1 and not frames: # if still no frames after trying many
                 break # Avoid infinite loop on problematic videos
        # If not enough frames, pad with last frame or black frames (simplistic padding)
        while len(frames) < num_frames_to_sample and len(frames) > 0:
            frames.append(frames[-1]) 
        while len(frames) < num_frames_to_sample: # If no frames at all
            # Create a black frame, assuming image_processor can handle it or get dimensions from config
            # This part might need image_size from config if not available otherwise
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8)) # Placeholder size
            print(f"Warning: Video has 0 frames or metadata issue. Using blank frames for {container.name}")

    else:
        total_frames = container.streams.video[0].frames
        indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)
        
        # Ensure indices are within the bounds of available frames
        indices = np.clip(indices, 0, total_frames - 1)

        frame_dict = {}
        container.seek(0) # Reset stream
        for i, frame in enumerate(container.decode(video=0)):
            if i > np.max(indices): # Optimization: stop decoding if all required frames are collected
                break
            if i in indices:
                frame_dict[i] = frame.to_ndarray(format="rgb24")
        
        # Fill frames in the order of indices, handling cases where some frames might not be decoded
        for idx in indices:
            if idx in frame_dict:
                frames.append(frame_dict[idx])
            elif frames: # If some frames already collected, use the last available one
                frames.append(frames[-1])
                print(f"Warning: Could not decode frame {idx}, using last available frame for {container.name}")
            else: # If no frames collected yet (e.g. first index fails)
                 # Create a black frame, assuming image_processor can handle it or get dimensions from config
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8)) # Placeholder size
                print(f"Warning: Could not decode frame {idx}, using blank frame for {container.name}")


    # Ensure exactly num_frames_to_sample are returned
    if not frames: # If absolutely no frames could be read/generated
        print(f"Error: Could not read any frames from {container.name}. Returning list of zeros.")
        # This requires knowing the expected frame dimensions if image_processor can't handle empty lists
        # For now, returning zeros based on a common size.
        # Get H, W from a typical config if possible, e.g. image_processor.size["shortest_edge"]
        # or model.config.image_size
        h, w = 224, 224 # Default fallback
        return [np.zeros((h, w, 3), dtype=np.uint8)] * num_frames_to_sample

    # Pad if fewer frames were decoded than requested
    while len(frames) < num_frames_to_sample:
        frames.append(frames[-1]) # Pad with the last frame
    
    return frames[:num_frames_to_sample]


def infer_with_bitsandbytes():
    """
    Performs video classification using the fine-tuned tiny VideoMAE model,
    quantized on-the-fly with bitsandbytes.
    """
    model_dir_original = "videomae-tiny-finetuned-xd-violence"
    absolute_model_path_original = os.path.abspath(model_dir_original)
    test_videos_dir = os.path.join("test", "videos")

    print("Note on the \'-pruned-quantized\' model directory:")
    print("The 'videomae-tiny-finetuned-xd-violence-pruned-quantized/pytorch_model.bin' file ")
    print("was created using PyTorch's `torch.quantization.quantize_dynamic` method. ")
    print("`bitsandbytes` is typically used by loading a model with `load_in_8bit=True` (or 4bit), ")
    print("which applies `bitsandbytes` quantization kernels to the original float model weights.")
    print("This script demonstrates the latter approach by loading the original fine-tuned tiny model ")
    print("and applying 8-bit quantization via `bitsandbytes` integration in Hugging Face transformers.\n")


    if not os.path.isdir(absolute_model_path_original):
        print(f"ERROR: Original model directory not found at {absolute_model_path_original}")
        print("Please ensure the 'videomae-tiny-finetuned-xd-violence' model directory exists.")
        return

    if not os.path.isdir(test_videos_dir):
        print(f"ERROR: Test videos directory not found at {test_videos_dir}")
        print("Please ensure the 'test/videos' directory exists and contains video files.")
        return

    video_files = glob.glob(os.path.join(test_videos_dir, "*.mp4")) # Add other extensions if needed
    if not video_files:
        print(f"No video files found in {test_videos_dir}")
        return
        
    print(f"Found {len(video_files)} videos in {test_videos_dir}")

    try:
        print(f"Loading image processor from: {absolute_model_path_original}")
        image_processor = VideoMAEImageProcessor.from_pretrained(absolute_model_path_original)
        
        print(f"Loading model from: {absolute_model_path_original} with 8-bit quantization (bitsandbytes)")
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16  # Specify compute dtype for bitsandbytes
        )
        
        model = VideoMAEForVideoClassification.from_pretrained(
            absolute_model_path_original,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,  # Set overall model dtype for non-quantized parts / activations
            device_map="auto"
            # num_channels=3 was removed, should be in config
        )
        model.eval() # Set model to evaluation mode
        
        # If not using device_map="auto" or want to specify a device for pixel_values later:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = model.to(device) # if load_in_8bit didn't handle it or no device_map
        print(f"Model loaded on device: {model.device}")
        
    except ImportError:
        print("ERROR: `bitsandbytes` or `accelerate` library not found. Please install them: `pip install bitsandbytes accelerate`")
        return
    except Exception as e:
        print(f"Error loading model or image processor: {e}")
        return

    num_frames_to_sample = model.config.num_frames
    id2label = model.config.id2label

    print("\nStarting inference...")
    for video_path in video_files:
        print(f"\nProcessing video: {video_path}")
        try:
            container = av.open(video_path)
            video_frames = read_video_pyav(container, num_frames_to_sample)
            container.close()

            if not video_frames:
                print(f"Could not read frames from {video_path}. Skipping.")
                continue

            inputs = image_processor(video_frames, return_tensors="pt")
            
            # Model expects pixel_values: (batch_size, num_frames, num_channels, height, width)
            # Image processor output is typically: (batch_size, num_channels, num_frames, height, width)
            # The model's forward pass should handle the permutation if needed.
            # pixel_values = inputs.pixel_values.permute(0, 2, 1, 3, 4) # Removed this manual permutation

            # Move inputs to the same device as the model if not handled by device_map
            # For models loaded with device_map="auto", inputs should also be on the correct device or moved.
            # If model is on cuda:0, pixel_values should be too.
            # If bitsandbytes loaded model to specific device, ensure input matches.
            # For simplicity, assuming model.device is the target.
            processed_pixel_values = inputs.pixel_values.to(model.device)

            # Cast pixel_values to the model's expected dtype (e.g., float16 if using 8-bit quantization)
            processed_pixel_values = processed_pixel_values.to(model.dtype)

            with torch.no_grad():
                outputs = model(pixel_values=processed_pixel_values) # Use direct processor output
                logits = outputs.logits

            # Aggregate scores by main category
            probabilities = torch.softmax(logits, dim=-1)[0] # Get probabilities for the first (and only) batch item
            aggregated_scores = {}

            for i, prob in enumerate(probabilities):
                original_label = id2label.get(i, f"Unknown ID: {i}")
                print(f"  Label: {original_label}, Score: {prob.item():.4f}")
                main_category = original_label.split('-')[0]
                score = prob.item()
                aggregated_scores[main_category] = aggregated_scores.get(main_category, 0.0) + score
            
            print("Aggregated Scores by Main Category:")
            if aggregated_scores:
                sorted_aggregated_scores = sorted(aggregated_scores.items(), key=lambda item: item[1], reverse=True)
                for category, summed_score in sorted_aggregated_scores:
                    print(f"  Main Category: {category}, Aggregated Score: {summed_score:.4f}")
            else:
                print("  Could not aggregate scores.")

        except av.error.AVError as ave:  # Changed to av.error.AVError
            print(f"PyAV Error processing {video_path}: {ave}. Skipping.")
        except Exception as e:
            print(f"An error occurred during classification of {video_path}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Ensure you have the necessary libraries:
    # pip install torch torchvision torchaudio transformers bitsandbytes accelerate av
    infer_with_bitsandbytes()

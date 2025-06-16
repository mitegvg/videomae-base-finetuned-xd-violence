import torch
from transformers import VideoMAEForVideoClassification, VideoMAEConfig

# Load config and model
config = VideoMAEConfig.from_pretrained("./public/")
config.do_resize_channels = True  # Add this line
model = VideoMAEForVideoClassification.from_pretrained(
    "./public/",
    config=config,
    trust_remote_code=True,
    local_files_only=True
)
model.eval()

# Verify channel count
print("Config num_channels:", model.config.num_channels)  # Should be 3

# Create only the dummy input—no processor, no image data
# In convert_safetensors_to_onnx.py
dummy_input = torch.randn(
    1, 
    model.config.num_frames,    # num_frames (T) at index 1
    model.config.num_channels,  # num_channels (C) at index 2
    model.config.image_size, 
    model.config.image_size, 
    dtype=torch.float32
)

# Wrap the model to enforce keyword-only input to tracing
class ExportableWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        # this forces tracer to use exactly this single keyword
        return self.m(pixel_values=pixel_values).logits

wrapped_model = ExportableWrapper(model)

# Export to ONNX
torch.onnx.export(
    wrapped_model,
    dummy_input,
    "videomae.onnx",
    input_names=['pixel_values'],  # Input name for the ONNX model
    output_names=['logits'],       # Output name for the ONNX model
    dynamic_axes={'pixel_values': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}, # Optional: if batch size can vary
    opset_version=16, # Changed from 11 to 16
    export_params=True,
    do_constant_folding=True,
)
print("✅ Successfully exported to videomae.onnx")

"""
Tri-Model Asymmetric Distillation Framework Models

This module implements the core models and framework for tri-model distillation,
including model loading, feature extraction, and integration with HuggingFace transformers.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import (
    VideoMAEForVideoClassification, 
    VideoMAEImageProcessor,
    VideoMAEConfig
)
import gdown
import logging

# Handle both relative and absolute imports
try:
    from .config import TriModelConfig, SSV2ModelConfig
except ImportError:
    from config import TriModelConfig, SSV2ModelConfig

logger = logging.getLogger(__name__)


class SSV2ModelLoader:
    """Utility class for downloading and loading SSV2 pretrained models from AMD MODEL_ZOO."""
    
    def __init__(self, config: SSV2ModelConfig):
        self.config = config
        os.makedirs(config.models_dir, exist_ok=True)
    
    def download_model(self, backbone: str = None, use_finetuned: bool = None) -> str:
        """
        Download SSV2 model from Google Drive if not already cached.
        
        Args:
            backbone: Model backbone ('ViT-S' or 'ViT-B')
            use_finetuned: Whether to use fine-tuned or pre-trained model
            
        Returns:
            Local path to the downloaded model
        """
        model_path = self.config.get_model_path(backbone, use_finetuned)
        
        if os.path.exists(model_path):
            logger.info(f"Model already exists at {model_path}")
            return model_path
        
        download_url = self.config.get_download_url(backbone, use_finetuned)
        
        # Extract file ID from Google Drive URL
        file_id = self._extract_file_id_from_url(download_url)
        
        logger.info(f"Downloading SSV2 model from {download_url}")
        try:
            gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
            logger.info(f"Successfully downloaded model to {model_path}")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            # Create a placeholder file to avoid repeated download attempts
            torch.save({}, model_path)
            logger.warning(f"Created placeholder file at {model_path}")
        
        return model_path
    
    def _extract_file_id_from_url(self, url: str) -> str:
        """Extract Google Drive file ID from sharing URL."""
        if "/file/d/" in url:
            return url.split("/file/d/")[1].split("/")[0]
        elif "id=" in url:
            return url.split("id=")[1].split("&")[0]
        else:
            raise ValueError(f"Cannot extract file ID from URL: {url}")
    
    def load_model(
        self, 
        backbone: str = None, 
        use_finetuned: bool = None,
        num_labels: int = 174  # SSV2 has 174 classes
    ) -> VideoMAEForVideoClassification:
        """
        Load SSV2 model with proper configuration.
        
        Args:
            backbone: Model backbone ('ViT-S' or 'ViT-B')
            use_finetuned: Whether to use fine-tuned or pre-trained model
            num_labels: Number of output classes
            
        Returns:
            Loaded VideoMAE model
        """
        model_path = self.download_model(backbone, use_finetuned)
        
        # Create model configuration based on backbone
        if backbone == "ViT-S":
            config = VideoMAEConfig(
                num_hidden_layers=12,
                hidden_size=384,
                intermediate_size=1536,
                num_attention_heads=6,
                num_labels=num_labels,
                image_size=224,
                num_frames=16
            )
        else:  # ViT-B
            config = VideoMAEConfig(
                num_hidden_layers=12,
                hidden_size=768,
                intermediate_size=3072,
                num_attention_heads=12,
                num_labels=num_labels,
                image_size=224,
                num_frames=16
            )
        
        # Initialize model
        model = VideoMAEForVideoClassification(config)
        
        # Load weights if available
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Handle potential key mismatches
            model_state_dict = model.state_dict()
            matched_state_dict = {}
            
            for key, value in state_dict.items():
                if key in model_state_dict and value.shape == model_state_dict[key].shape:
                    matched_state_dict[key] = value
                else:
                    logger.warning(f"Skipping key {key} due to shape mismatch or missing key")
            
            model.load_state_dict(matched_state_dict, strict=False)
            logger.info(f"Loaded SSV2 model weights from {model_path}")
            
        except Exception as e:
            logger.warning(f"Could not load weights from {model_path}: {e}")
            logger.info("Using randomly initialized model")
        
        return model


class TriModelDistillationFramework(nn.Module):
    """
    Main framework class that orchestrates the tri-model distillation process.
    
    Manages:
    1. Teacher model (VideoMAE base, frozen)
    2. Assistant model (SSV2 pretrained, frozen)  
    3. Student model (target for fine-tuning)
    """
    
    def __init__(
        self,
        config: TriModelConfig,
        num_labels: int,
        label2id: Dict[str, int],
        id2label: Dict[int, str]
    ):
        super().__init__()
        
        self.config = config
        self.num_labels = num_labels
        self.label2id = label2id
        self.id2label = id2label
        
        # Initialize models
        self._initialize_models()
        
        # Freeze teacher and assistant models
        self._freeze_model(self.teacher_model)
        if hasattr(self, 'assistant_model'):
            self._freeze_model(self.assistant_model)
    
    def _initialize_models(self):
        """Initialize teacher, assistant, and student models."""
        
        # 1. Initialize Teacher Model (VideoMAE base)
        logger.info("Loading teacher model...")
        self.teacher_model = VideoMAEForVideoClassification.from_pretrained(
            self.config.teacher_model_name,
            num_labels=self.num_labels,
            label2id=self.label2id,
            id2label=self.id2label,
            ignore_mismatched_sizes=True
        )
        self.teacher_model.config.mask_ratio = 0.0
        
        # 2. Initialize Assistant Model (SSV2)
        logger.info("Loading assistant model...")
        if self.config.assistant_model_path:
            ssv2_config = SSV2ModelConfig()
            ssv2_loader = SSV2ModelLoader(ssv2_config)
            
            try:
                self.assistant_model = ssv2_loader.load_model(
                    backbone="ViT-B", 
                    use_finetuned=True,
                    num_labels=174  # SSV2 original classes
                )
                
                # Adapt assistant model to target domain
                self._adapt_assistant_model()
                
            except Exception as e:
                logger.warning(f"Could not load assistant model: {e}")
                logger.info("Using teacher model as assistant (fallback)")
                self.assistant_model = VideoMAEForVideoClassification.from_pretrained(
                    self.config.teacher_model_name,
                    num_labels=self.num_labels,
                    label2id=self.label2id,
                    id2label=self.id2label,
                    ignore_mismatched_sizes=True
                )
        else:
            logger.info("No assistant model path provided, using teacher as assistant")
            self.assistant_model = self.teacher_model
        
        # 3. Initialize Student Model (using configurable tiny model if specified)
        logger.info("Loading student model...")
        
        # Temporarily add required attributes to config for student model creation
        original_num_labels = getattr(self.config, 'num_labels', None)
        original_label2id = getattr(self.config, 'label2id', None)
        original_id2label = getattr(self.config, 'id2label', None)
        
        # Set the required attributes
        self.config.num_labels = self.num_labels
        self.config.label2id = self.label2id
        self.config.id2label = self.id2label
        
        # Load student model using our custom function
        self.student_model = load_student_model(config=self.config)
        
        # Restore original config attributes if they existed
        if original_num_labels is not None:
            self.config.num_labels = original_num_labels
        if original_label2id is not None:
            self.config.label2id = original_label2id
        if original_id2label is not None:
            self.config.id2label = original_id2label
        
        self.student_model.config.mask_ratio = self.config.mask_ratio
        
        # Initialize image processor
        self.image_processor = VideoMAEImageProcessor.from_pretrained(
            self.config.teacher_model_name
        )
    
    def _adapt_assistant_model(self):
        """Adapt assistant model's classifier to target domain."""
        if hasattr(self.assistant_model, 'classifier'):
            # Replace classifier head to match target number of classes
            in_features = self.assistant_model.classifier.in_features
            self.assistant_model.classifier = nn.Linear(in_features, self.num_labels)
            
            # Initialize with small random weights
            nn.init.normal_(self.assistant_model.classifier.weight, std=0.02)
            nn.init.zeros_(self.assistant_model.classifier.bias)
    
    def _freeze_model(self, model: nn.Module):
        """Freeze all parameters of a model."""
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    
    def forward(
        self, 
        pixel_values: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        output_attentions: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass through all three models.
        
        Args:
            pixel_values: Input video tensor
            labels: Ground truth labels (optional)
            output_hidden_states: Whether to output hidden states
            output_attentions: Whether to output attention weights
            
        Returns:
            Dictionary containing outputs from all three models
        """
        
        # Student forward pass (with gradients)
        student_outputs = self.student_model(
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                pixel_values=pixel_values,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions
            )
        
        # Assistant forward pass (no gradients)
        with torch.no_grad():
            assistant_outputs = self.assistant_model(
                pixel_values=pixel_values,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions
            )
        
        return {
            'student': student_outputs,
            'teacher': teacher_outputs,
            'assistant': assistant_outputs
        }
    
    def get_student_model(self) -> VideoMAEForVideoClassification:
        """Get the student model for training/inference."""
        return self.student_model
    
    def save_student_model(self, save_path: str):
        """Save the fine-tuned student model."""
        self.student_model.save_pretrained(save_path)
        self.image_processor.save_pretrained(save_path)
        logger.info(f"Student model saved to {save_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all models in the framework."""
        return {
            'teacher': {
                'name': self.config.teacher_model_name,
                'num_parameters': sum(p.numel() for p in self.teacher_model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.teacher_model.parameters() if p.requires_grad)
            },
            'assistant': {
                'name': 'SSV2_pretrained' if hasattr(self, 'assistant_model') else 'teacher_fallback',
                'num_parameters': sum(p.numel() for p in self.assistant_model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.assistant_model.parameters() if p.requires_grad)
            },
            'student': {
                'name': self.config.student_model_name,
                'num_parameters': sum(p.numel() for p in self.student_model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
            }
        }


# Helper functions for notebook compatibility
def load_teacher_model(config: TriModelConfig) -> VideoMAEForVideoClassification:
    """
    Load and initialize the teacher model.
    
    Args:
        config: Tri-model configuration
        
    Returns:
        Initialized teacher model
    """
    logger.info(f"Loading teacher model: {config.teacher_model_name}")
    
    model = VideoMAEForVideoClassification.from_pretrained(
        config.teacher_model_name,
        num_labels=config.num_labels if hasattr(config, 'num_labels') else 2,
        ignore_mismatched_sizes=True
    )
    
    # Freeze the model
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    logger.info("Teacher model loaded and frozen")
    return model


def load_assistant_model(ssv2_config: SSV2ModelConfig, tri_config: TriModelConfig) -> VideoMAEForVideoClassification:
    """
    Load and initialize the assistant model.
    
    Args:
        ssv2_config: SSV2 model configuration
        tri_config: Tri-model configuration
        
    Returns:
        Initialized assistant model
    """
    logger.info("Loading assistant model from SSV2")
    
    try:
        ssv2_loader = SSV2ModelLoader(ssv2_config)
        model = ssv2_loader.load_model(
            backbone="ViT-B",
            use_finetuned=True,
            num_labels=ssv2_config.num_classes
        )
        
        # Adapt classifier if needed
        if hasattr(tri_config, 'num_labels') and tri_config.num_labels != ssv2_config.num_classes:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, tri_config.num_labels)
            nn.init.normal_(model.classifier.weight, std=0.02)
            nn.init.zeros_(model.classifier.bias)
        
    except Exception as e:
        logger.warning(f"Could not load SSV2 assistant model: {e}")
        logger.info("Falling back to teacher model as assistant")
        
        # Fallback to teacher model
        model = VideoMAEForVideoClassification.from_pretrained(
            tri_config.teacher_model_name,
            num_labels=tri_config.num_labels if hasattr(tri_config, 'num_labels') else 2,
            ignore_mismatched_sizes=True
        )
    
    # Freeze the model
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    logger.info("Assistant model loaded and frozen")
    return model


def load_student_model(config: TriModelConfig) -> VideoMAEForVideoClassification:
    """
    Load and initialize the student model as a smaller/tiny version.
    
    Args:
        config: Tri-model configuration
        
    Returns:
        Initialized student model (trainable, smaller architecture)
    """
    logger.info(f"Loading student model: {config.student_model_name}")
    
    # Create a smaller student model configuration
    from transformers import VideoMAEConfig
    
    # Check if we want a tiny student model
    use_tiny_student = getattr(config, 'use_tiny_student', True)
    student_num_layers = getattr(config, 'student_num_layers', 4)  # Default to 4 layers
    
    if use_tiny_student:
        logger.info(f"Creating tiny student model with {student_num_layers} layers")
        
        # Load base configuration and modify it for smaller model
        base_config = VideoMAEConfig.from_pretrained(config.student_model_name)
        
        # Get student architecture parameters from config
        student_hidden_size = getattr(config, 'student_hidden_size', 384)
        student_num_attention_heads = getattr(config, 'student_num_attention_heads', 6)
        
        # Create tiny model configuration
        tiny_config = VideoMAEConfig(
            # Core architecture - much smaller
            hidden_size=student_hidden_size,  # Configurable hidden size
            intermediate_size=student_hidden_size * 4,  # 4x hidden size as typical
            num_hidden_layers=student_num_layers,  # Much fewer layers
            num_attention_heads=student_num_attention_heads,  # Configurable attention heads
            
            # Keep video-specific parameters
            image_size=base_config.image_size,
            patch_size=base_config.patch_size,
            num_channels=base_config.num_channels,
            tubelet_size=base_config.tubelet_size,
            num_frames=base_config.num_frames,
            
            # Classification head
            num_labels=config.num_labels if hasattr(config, 'num_labels') else 2,
            
            # Other parameters
            hidden_act=base_config.hidden_act,
            hidden_dropout_prob=0.1,  # Slightly higher dropout for regularization
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=base_config.layer_norm_eps,
            qkv_bias=base_config.qkv_bias,
        )
        
        # Create model with tiny configuration
        model = VideoMAEForVideoClassification(tiny_config)
        
        # Initialize weights properly
        model.apply(model._init_weights)
        
        # Log model size comparison
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Tiny student model created:")
        logger.info(f"  - Layers: {student_num_layers} (vs 12 in teacher)")
        logger.info(f"  - Hidden size: {student_hidden_size} (vs 768 in teacher)")
        logger.info(f"  - Attention heads: {student_num_attention_heads} (vs 12 in teacher)")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Parameter reduction: ~{((87000000 - total_params) / 87000000 * 100):.1f}% vs full VideoMAE")
        
    else:
        # Load full-size student model (original behavior)
        model = VideoMAEForVideoClassification.from_pretrained(
            config.student_model_name,
            num_labels=config.num_labels if hasattr(config, 'num_labels') else 2,
            ignore_mismatched_sizes=True
        )
        logger.info("Full-size student model loaded")
    
    # Student model should be trainable
    model.train()
    
    logger.info("Student model loaded (trainable)")
    return model


# Alias for backward compatibility
TriModelOrchestrator = TriModelDistillationFramework

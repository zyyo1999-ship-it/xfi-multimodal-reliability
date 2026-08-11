"""Checkpoint-only X-Fi construction for inference and robustness studies.

The official constructor loads four modality-specific backbone checkpoints and
then the complete X-Fi checkpoint overwrites every parameter. This module
builds the identical module tree without those redundant preliminary loads.
Strict loading of the complete checkpoint remains mandatory in the runner.
"""

from __future__ import annotations

import os

import torch
from torch import nn


MMWAVE_MODALITY = (False, False, True, False)
MMWAVE_ONLY_SPECIALIZATION = "xfi-mmfi-har-mmwave-only-inference-v1"

# These modules are never reached when X-Fi receives the fixed mmWave-only
# modality mask. Removing their tensors changes neither the executed graph nor
# its outputs, but the resulting checkpoint must not be used for other masks.
MMWAVE_ONLY_INACTIVE_PREFIXES = (
    "feature_extractor.rgb_extractor.",
    "feature_extractor.depth_extractor.",
    "feature_extractor.lidar_extractor.",
    "linear_projector.rgb_linear_projection.",
    "linear_projector.depth_linear_projection.",
    "linear_projector.lidar_linear_projection.",
    "linear_projector.pos_enc_layer.",
    "X_Fusion_block.kv_layers.0.",
    "X_Fusion_block.kv_layers.1.",
    "X_Fusion_block.kv_layers.3.",
    "X_Fusion_block.cross_attention_transformer.transformer_layers.0.",
    "X_Fusion_block.cross_attention_transformer.transformer_layers.1.",
    "X_Fusion_block.cross_attention_transformer.transformer_layers.3.",
)


def is_mmwave_inactive_key(key: str) -> bool:
    """Return whether a parameter is unreachable under the mmWave-only mask."""
    return key.startswith(MMWAVE_ONLY_INACTIVE_PREFIXES)


def require_multimodal_checkpoint(metadata: dict) -> None:
    """Reject a checkpoint whose omitted branches are needed by Track B."""
    if metadata.get("specialization") == MMWAVE_ONLY_SPECIALIZATION:
        raise RuntimeError(
            "The multimodal study requires the complete X-Fi checkpoint; the "
            "mmWave-only specialization has no valid LiDAR branch."
        )


def load_xfi_state_dict(
    model: nn.Module,
    checkpoint_payload: dict,
) -> dict:
    """Load either the official full state dict or our verified specialization.

    A specialized checkpoint is accepted only when every missing model tensor
    belongs to an explicitly unreachable mmWave-only branch. This prevents a
    damaged or accidentally incomplete checkpoint from being silently used.
    """
    if "state_dict" in checkpoint_payload:
        state_dict = checkpoint_payload["state_dict"]
        metadata = dict(checkpoint_payload.get("metadata", {}))
    else:
        state_dict = checkpoint_payload
        metadata = {"specialization": "official-full-checkpoint"}

    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected checkpoint keys: {load_result.unexpected_keys}"
        )

    missing = set(load_result.missing_keys)
    if not missing:
        return metadata

    model_keys = set(model.state_dict())
    expected_inactive = {
        key for key in model.state_dict() if is_mmwave_inactive_key(key)
    }
    if metadata.get("specialization") != MMWAVE_ONLY_SPECIALIZATION:
        raise RuntimeError(
            "An incomplete checkpoint must declare the verified mmWave-only "
            "specialization."
        )
    active_missing = sorted((model_keys - set(state_dict)) - expected_inactive)
    reported_active_missing = sorted(missing - expected_inactive)
    if active_missing or reported_active_missing:
        raise RuntimeError(
            "Specialized checkpoint key mismatch: "
            f"active_missing={active_missing}, "
            f"reported_active_missing={reported_active_missing}"
        )
    return metadata


def build_xfi_for_checkpoint(model_depth: int = 2) -> nn.Module:
    """Construct an X-Fi model whose parameters must come from a full checkpoint."""
    import X_Fi as official

    class CheckpointFeatureExtractor(official.feature_extrator):
        def __init__(self) -> None:
            nn.Module.__init__(self)

            rgb_model = official.RGB_ResNet18()
            depth_model = official.Depth_ResNet18()
            mmwave_model = official.mmwave_PointTransformerReg()
            lidar_model = official.lidar_PointTransformer_cls(root=os.getcwd())

            self.rgb_extractor = official.rgb_feature_extractor(rgb_model)
            self.depth_extractor = official.depth_feature_extractor(depth_model)
            self.mmwave_extractor = official.mmwave_feature_extractor(mmwave_model)
            self.lidar_extractor = official.lidar_feature_extractor(lidar_model)

    class CheckpointXFi(official.X_Fi):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.feature_extractor = CheckpointFeatureExtractor()
            self.linear_projector = official.linear_projector(512, 512)
            self.X_Fusion_block = official.X_Fusion(
                num_modalities=4,
                dim=512,
                qkv_hidden_expansion=2,
                hidden_dim=512,
                num_feature=32,
                num_heads=8,
                dim_heads=64,
                model_depth=model_depth,
                dropout=0.0,
            )

    return CheckpointXFi()

"""FLUX model specifications for the text-to-image capability.

``schnell`` is the ungated, distilled default (~4 steps, guidance 0, no negative
prompt). ``dev`` is the gated, higher-fidelity option (~20-50 steps, real
guidance, optional negative prompt via true CFG).
"""

from __future__ import annotations

from localai.core.registry import ModelSpec

CAPABILITY_ID = "text-to-image"

SCHNELL = ModelSpec(
    model_id="schnell",
    capability_id=CAPABILITY_ID,
    repo="black-forest-labs/FLUX.1-schnell",
    pipeline_class="FluxPipeline",
    display_name="FLUX.1-schnell (fast, ungated)",
    default_steps=4,
    min_steps=1,
    max_steps=12,
    supports_guidance=False,
    default_guidance=0.0,
    supports_negative_prompt=False,
    gated=False,
    default_width=1024,
    default_height=1024,
    size_multiple=16,
    max_sequence_length=256,
    recommended_dtype="bfloat16",
    is_default=True,
    notes="Apache-2.0, distilled for ~4 steps; more steps do NOT improve quality.",
)

DEV = ModelSpec(
    model_id="dev",
    capability_id=CAPABILITY_ID,
    repo="black-forest-labs/FLUX.1-dev",
    pipeline_class="FluxPipeline",
    display_name="FLUX.1-dev (higher fidelity, gated)",
    default_steps=28,
    min_steps=1,
    max_steps=50,
    supports_guidance=True,
    default_guidance=3.5,
    supports_negative_prompt=True,
    gated=True,
    default_width=1024,
    default_height=1024,
    size_multiple=16,
    max_sequence_length=512,
    recommended_dtype="bfloat16",
    is_default=False,
    notes="Gated non-commercial license; requires accepting the license + HF token.",
)

MODELS = [SCHNELL, DEV]

"""Narrow model-format adapters for the pinned native math runtime."""


def gemma3_rollout_weight_name(name):
    """Restore SigLIP's checkpoint wrapper removed by Transformers 5.

    vLLM's Gemma mapper handles the outer ``model.`` prefix, but its SigLIP
    loader still expects ``vision_model`` below ``vision_tower``. Only names
    change; the complete multimodal actor and every tensor are retained.
    """
    for prefix in ("model.vision_tower.", "vision_tower."):
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            if suffix.startswith("vision_model.vision_model."):
                raise ValueError(f"Ambiguous Gemma vision weight name: {name}")
            if not suffix.startswith("vision_model."):
                return prefix + "vision_model." + suffix
    return name


def gemma3_rollout_weights(weights):
    seen = set()
    for name, tensor in weights:
        name = gemma3_rollout_weight_name(name)
        if name in seen:
            raise ValueError(f"Duplicate Gemma rollout weight: {name}")
        seen.add(name)
        yield name, tensor

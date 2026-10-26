"""Weight synchronization must preserve all tensors across HF/vLLM layouts."""

import pytest

from rllab.math_model_compat import gemma3_rollout_weight_name, gemma3_rollout_weights


def test_gemma_names_preserve_language_weights_and_already_wrapped_vision():
    tensor = object()
    pairs = [
        ("model.vision_tower.embeddings.patch_embedding.weight", tensor),
        ("model.vision_tower.encoder.layers.0.self_attn.q_proj.weight", tensor),
        ("model.language_model.layers.0.self_attn.q_proj.weight", tensor),
        ("lm_head.weight", tensor),
        ("vision_tower.vision_model.post_layernorm.weight", tensor),
    ]
    result = list(gemma3_rollout_weights(iter(pairs)))
    assert [name for name, _ in result] == [
        "model.vision_tower.vision_model.embeddings.patch_embedding.weight",
        "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
        pairs[2][0], pairs[3][0], pairs[4][0],
    ]
    assert all(value is tensor for _, value in result)
    assert list(gemma3_rollout_weights(result)) == result


def test_gemma_names_reject_collisions_and_double_wrappers():
    with pytest.raises(ValueError, match="Duplicate"):
        list(gemma3_rollout_weights([
            ("model.vision_tower.embeddings.patch_embedding.weight", object()),
            ("model.vision_tower.vision_model.embeddings.patch_embedding.weight", object()),
        ]))
    with pytest.raises(ValueError, match="Ambiguous"):
        gemma3_rollout_weight_name("vision_tower.vision_model.vision_model.x")

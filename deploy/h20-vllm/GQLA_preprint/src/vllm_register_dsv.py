"""Register the DeepSeek-V2/V3 GQLA model classes with vLLM.

Two entry points to the same ``register()`` call:

* **Import side-effect** — ``import src.vllm_register_dsv`` registers in
  the current process. Sufficient for ``tensor_parallel_size=1``.
* **vLLM plugin entry point** — ``pyproject.toml`` exposes ``register``
  under ``vllm.general_plugins``. After ``pip install -e .`` vLLM calls
  it in every process (main + each TP worker), enabling
  ``tensor_parallel_size >= 2``.
"""

from vllm import ModelRegistry

_REGISTERED = False


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    ModelRegistry.register_model(
        "DeepseekV3GQLAForCausalLM",
        "src.vllm_model_dsv3:DeepseekV3GQLAForCausalLM",
    )
    ModelRegistry.register_model(
        "DeepseekV3GQLAGQAForCausalLM",
        "src.vllm_model_dsv3:DeepseekV3GQLAGQAForCausalLM",
    )
    ModelRegistry.register_model(
        "DeepseekV3GQLAHPCForCausalLM",
        "src.vllm_model_dsv3:DeepseekV3GQLAHPCForCausalLM",
    )
    ModelRegistry.register_model(
        "DeepseekV2GQLAForCausalLM",
        "src.vllm_model_dsv3:DeepseekV2GQLAForCausalLM",
    )
    _REGISTERED = True


register()

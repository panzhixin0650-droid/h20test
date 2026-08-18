"""Minimal GQLA vLLM serving plugin.

The deployment bundle intentionally omits conversion and training modules.
Model classes are imported lazily through the vLLM registry entry points.
"""

__all__: list[str] = []

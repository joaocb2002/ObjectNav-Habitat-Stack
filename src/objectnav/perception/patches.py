"""
Perception patches (Ultralytics monkey-patching).

We keep legacy patch code intact and call it explicitly from one place so:
- imports are controlled
- we can later refactor safely
"""

def apply_ultralytics_patch() -> None:
    # Importing this module applies the monkey patches at import time.
    # Keep this import inside the function so nothing happens accidentally.
    from objectnav.legacy.mylib import yolo_patch_softmax  # noqa: F401

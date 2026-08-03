from .layout import install as install_layout_patch
from .nodes import MiniMaxH3MultiFrameExtension


class Extension(MiniMaxH3MultiFrameExtension):
    async def on_load(self):
        install_layout_patch()


async def comfy_entrypoint() -> Extension:
    return Extension()

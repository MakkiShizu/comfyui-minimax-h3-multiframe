"""Generalize PackedLayout's keyframe anchors to arbitrary frame indices.

comfy/ldm/minimax/model.py accepts index 0 and frame_count - 1 only and raises
for anything else. That check is a policy, not an architectural limit: the rope
t coordinate of a keyframe is text_len + FRAME_RESCALE * index for every index,
and nothing else in the packed layout or the DiT depends on where a cond row
sits (see grid.py for the derivation).

The patch keeps core's own construction: it builds the layout with all anchors
stubbed to index 0, which yields exactly one cond segment per keyframe in
keyframe order, then rewrites the t column of those segments. A layout whose
anchors are all first/last is handed to core untouched, so stock workflows keep
their exact numbers.

Both call sites resolve the class at call time - model_base.py through
``comfy.ldm.minimax.model.PackedLayout`` and the DiT through its own module
global - so replacing the module attribute covers them.
"""

import comfy.ldm.minimax.model as h3


def install():
    if getattr(h3.PackedLayout, "minimax_multiframe", False):
        return

    core_layout = h3.PackedLayout

    class MultiFramePackedLayout(core_layout):
        minimax_multiframe = True

        def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t,
                     keyframes=None, refs=None, frame_count=None):
            indices = [kf["resolved_frame_index"] for kf in keyframes] if keyframes else []
            last = frame_count - 1 if frame_count is not None else None
            if all(i == 0 or i == last for i in indices):
                super().__init__(text_len, latent_t, latent_h, latent_w, audio_t,
                                 keyframes=keyframes, refs=refs, frame_count=frame_count)
                return

            super().__init__(text_len, latent_t, latent_h, latent_w, audio_t,
                             keyframes=[{"resolved_frame_index": 0} for _ in indices],
                             refs=refs, frame_count=frame_count)

            cond = [(a, b) for a, b, kind in self.segments if kind == "cond"]
            if len(cond) != len(indices):
                raise RuntimeError("MiniMax H3 layout patch: expected one cond segment per keyframe, "
                                   "got {} for {} keyframes".format(len(cond), len(indices)))
            for (a, b), index in zip(cond, indices):
                self.position_ids[a:b, 0] = text_len + h3.FRAME_RESCALE * index

    h3.PackedLayout = MultiFramePackedLayout

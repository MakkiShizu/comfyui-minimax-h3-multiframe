"""Generalize PackedLayout's keyframe anchors to arbitrary frame indices.

comfy/ldm/minimax/model.py accepts index 0 and frame_count - 1 only and raises
for anything else. That check is a policy, not an architectural limit: the rope
t coordinate of a keyframe is text_len + FRAME_RESCALE * index for every index,
and nothing else in the packed layout or the DiT depends on where a cond row
sits (see grid.py for the derivation).

The patch builds the layout with every anchor stubbed to index 0, which every
shipped core version accepts (it yields exactly one cond segment per keyframe in
keyframe order), then rewrites the t column of those segments to the real
frame index. This is numerically identical to core for the first/last anchors
(core's last-frame t is text_len + sum(_video_t_spans(latent_t)) - FRAME_RESCALE
== text_len + FRAME_RESCALE * (frame_count - 1)) and works regardless of whether
the installed core build exposes the optional `frame_count` argument - so we
never pass it and never trip the first/last-only policy.

Both call sites resolve the class at call time - model_base.py through
``comfy.ldm.minimax.model.PackedLayout`` and the DiT through its own module
global - so replacing the module attribute covers them.
"""

import inspect

import comfy.ldm.minimax.model as h3


def install():
    if getattr(h3.PackedLayout, "minimax_multiframe", False):
        return

    core_layout = h3.PackedLayout
    core_accepts_frame_count = "frame_count" in inspect.signature(core_layout.__init__).parameters

    class MultiFramePackedLayout(core_layout):
        minimax_multiframe = True

        def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t,
                     keyframes=None, refs=None, frame_count=None):
            if not keyframes:
                super().__init__(text_len, latent_t, latent_h, latent_w, audio_t,
                                 keyframes=keyframes, refs=refs,
                                 **({"frame_count": frame_count} if core_accepts_frame_count else {}))
                return

            # Stub every anchor to index 0 so core builds one cond segment per
            # keyframe without hitting its first/last-only policy. The real rope
            # t is rewritten below, so the stub value never reaches the model.
            stubbed = [dict(kf, resolved_frame_index=0) for kf in keyframes]
            super().__init__(text_len, latent_t, latent_h, latent_w, audio_t,
                             keyframes=stubbed, refs=refs,
                             **({"frame_count": frame_count} if core_accepts_frame_count else {}))

            cond = [(a, b) for a, b, kind in self.segments if kind == "cond"]
            if len(cond) != len(keyframes):
                raise RuntimeError("MiniMax H3 layout patch: expected one cond segment per keyframe, "
                                   "got {} for {} keyframes".format(len(cond), len(keyframes)))
            for (a, b), kf in zip(cond, keyframes):
                index = kf["resolved_frame_index"]
                self.position_ids[a:b, 0] = text_len + h3.FRAME_RESCALE * index

    h3.PackedLayout = MultiFramePackedLayout

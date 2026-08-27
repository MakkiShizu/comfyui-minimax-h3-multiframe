"""MiniMax H3 conditioning on the first N and/or last N frames.

Stock MiniMaxH3ImageToVideo binds one image to frame 0 and one to the final
frame. This node binds two frame *runs* instead: the head run is anchored at
pixel frame 0, the tail run at frame_count - len(tail). Each run is encoded as
one clip by the video VAE and then split into its latent tokens, one keyframe
entry per token, so the DiT receives the same never-denoised cond rows it
already uses for single keyframes - only more of them, at the rope positions
their frames occupy.

Run lengths snap down to the VAE's grid (1, 5, 22, 39, ... frames). A tail run
of 17j+5 frames starts at 17(m-j), which is where a target latent token starts,
so the cond rows land exactly on the timeline positions they describe.

Pinning (pin_frames, default on): beyond the soft keyframe conditioning, the node
also emits a 'pin' spec listing the latent-token slots that each keyframe occupies
together with the clean keyframe latent. Wire that spec through
MiniMaxH3ApplyFramePin AFTER the sampler: it splices the clean latents back into
the sampled latent at those slots, so the decoded video shows the exact input
images at the pinned frames. This is a post-sampling operation and needs no
retraining; it is the reliable way to truly freeze frames, because the DiT only
receives keyframes as soft (never-denoised) context and never copies them into the
output.
"""

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers
import nodes
from comfy_api.latest import ComfyExtension, io

from .grid import (snap_condition_frames, temporal_shape, token_frame_offsets,
                   build_pin_spec)

LATENT_CHANNELS = 24
AUDIO_LATENT_CHANNELS = 32
SPATIAL_COMPRESSION = 16


class PinSpec(io.Custom("minimax_pin_spec")):
    """(token_slot, keyframe_latent[1,24,1,h,w]) pairs to splice into the sampled latent."""
    Type = list


def _resize(image, width, height, crop):
    # image [B, H, W, C] -> [B, height, width, 3]
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _empty_av_latent(width, height, length):
    frame_count, latent_t, audio_t = temporal_shape(length)
    device = comfy.model_management.intermediate_device()
    video = torch.zeros([1, LATENT_CHANNELS, latent_t,
                         height // SPATIAL_COMPRESSION, width // SPATIAL_COMPRESSION], device=device)
    audio = torch.zeros([1, AUDIO_LATENT_CHANNELS, 2, audio_t], device=device)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def _take_run(image, width, height, crop, from_end):
    n = snap_condition_frames(image.shape[0])
    frames = image[-n:] if from_end else image[:n]
    return _resize(frames, width, height, crop)


def _run_keyframes(vae, frames, start_index):
    z = vae.encode(frames)  # [1, 24, T, H/16, W/16]
    return [{"resolved_frame_index": start_index + offset, "latent": z[:, :, k:k + 1]}
            for k, offset in enumerate(token_frame_offsets(z.shape[2]))]


class MiniMaxH3MultiFrameToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MultiFrameToVideo",
            display_name="MiniMax H3 Multi-Frame to Video",
            category="model/conditioning/minimax",
            description="First-N / last-N frame conditioning for MiniMax H3. Feeding a single image to "
                        "each input reproduces MiniMaxH3ImageToVideo exactly; longer runs anchor a whole "
                        "head or tail segment. The released FL2VA weights were trained on first/last "
                        "keyframes only, so runs longer than one frame are outside the training set. "
                        "Enable 'pin_frames' and wire the 'pin' output through MiniMaxH3ApplyFramePin "
                        "(after the sampler) to freeze the conditioned frames to the exact input images.",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                             tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid "
                                     "(124 = ~5s; trained range is ~124-362, longer is untested)"),
                io.Combo.Input("prompt_frames", options=["anchor", "boundary", "none"], default="anchor",
                               tooltip="Which frames Qwen sees as <Picture i>. 'anchor' shows the first frame "
                                       "of the head run and the last frame of the tail run, matching the stock "
                                       "first/last presentation. 'boundary' shows the two frames next to the "
                                       "generated span instead. 'none' leaves the prompt text-only."),
                io.Float.Input("cond_noise_aug", default=0.999, min=0.0, max=1.0, step=0.001,
                               tooltip="Timestep the keyframe rows are pinned at. 0.999 is the model default; "
                                       "lower values noise the conditioning down and loosen its grip."),
                io.Boolean.Input("pin_frames", default=True, label_on="pinned", label_off="soft only",
                                 tooltip="When on, also emit a 'pin' spec so MiniMaxH3ApplyFramePin can freeze "
                                         "the conditioned frames to the exact input images after sampling. Off = "
                                         "soft keyframe conditioning only (frames are guided but not guaranteed "
                                         "pixel-identical)."),
                io.Image.Input("first_frames", optional=True,
                               tooltip="Run starting at frame 0, in order. Truncated to the VAE grid "
                                       "(1, 5, 22, 39, ... frames)."),
                io.Image.Input("last_frames", optional=True,
                               tooltip="Run ending on the final frame, in order. The last N frames are kept, "
                                       "truncated to the VAE grid (1, 5, 22, 39, ... frames)."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"),
                     io.Latent.Output(),
                     PinSpec.Output(display_name="pin")],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, width, height, length, prompt_frames, cond_noise_aug,
                pin_frames, first_frames=None, last_frames=None) -> io.NodeOutput:
        _, latent_t, _ = temporal_shape(length)
        latent, frame_count = _empty_av_latent(width, height, length)

        # head is the geometry anchor and gets a plain stretch, the tail follows with a cover-crop
        head = None if first_frames is None else _take_run(first_frames, width, height, "disabled", False)
        tail = None if last_frames is None else _take_run(last_frames, width, height, "center", True)

        head_n = 0 if head is None else head.shape[0]
        tail_n = 0 if tail is None else tail.shape[0]
        if head_n + tail_n > frame_count:
            raise ValueError("MiniMax H3: {} head + {} tail conditioning frames do not fit in a {} frame "
                             "clip, raise length or shorten a run".format(head_n, tail_n, frame_count))

        keyframes = []
        images = []
        if head is not None:
            keyframes += _run_keyframes(vae, head, 0)
            images.append(head[-1:] if prompt_frames == "boundary" else head[:1])
        if tail is not None:
            keyframes += _run_keyframes(vae, tail, frame_count - tail_n)
            images.append(tail[:1] if prompt_frames == "boundary" else tail[-1:])
        if prompt_frames == "none":
            images = []

        tokens = clip.tokenize(prompt, images=images)
        cond = clip.encode_from_tokens_scheduled(tokens)

        pin = build_pin_spec(keyframes, latent_t) if (pin_frames and keyframes) else []

        if keyframes:
            cond = node_helpers.conditioning_set_values(cond, {
                "minimax_keyframes": keyframes,
                "minimax_frame_count": frame_count,
                "minimax_visual_cond_noise_aug": cond_noise_aug,
            })
        return io.NodeOutput(cond, latent, pin)


class MiniMaxH3ApplyFramePin(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ApplyFramePin",
            display_name="MiniMax H3 Apply Frame Pin",
            category="model/conditioning/minimax",
            description="Splice the clean keyframe latents back into the sampled latent at the pinned "
                        "token slots, so those frames decode to the exact input images. Place this AFTER "
                        "the sampler and BEFORE VAE Decode. Connect its 'pin' input to the 'pin' output of "
                        "MiniMaxH3MultiFrameToVideo. A no-op when the pin spec is empty (pin_frames off).",
            inputs=[
                io.Latent.Input("latent", tooltip="Sampled latent from the MiniMax H3 sampler."),
                PinSpec.Input("pin", tooltip="Pin spec from MiniMaxH3MultiFrameToVideo."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, latent, pin) -> io.NodeOutput:
        samples = latent["samples"]
        is_nested = isinstance(samples, comfy.nested_tensor.NestedTensor)
        video = samples.tensors[0] if is_nested else samples
        if pin:
            video = video.clone()
            for slot, kf in pin:
                # kf: [1, 24, 1, h, w] -> place its single latent token into video[:, :, slot]
                video[0, :, slot, :, :] = kf[0, :, 0, :, :].to(device=video.device, dtype=video.dtype)
            if is_nested:
                samples = comfy.nested_tensor.NestedTensor((video, samples.tensors[1]))
            else:
                samples = video
        return io.NodeOutput({"samples": samples})


class MiniMaxH3MultiFrameExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3MultiFrameToVideo, MiniMaxH3ApplyFramePin]

"""MiniMax H3 temporal grid arithmetic.

The video VAE maps 17k+5 pixel frames to 5k+2 latent tokens: token 0 covers a
single pixel frame, every following token covers 4, cycling through
FRAME_PER_TOKEN = (1, 4, 4, 4, 4). The DiT's rope t axis is that same partition
scaled by FRAME_RESCALE, so a latent token starting at pixel frame p sits at
t = text_len + FRAME_RESCALE * p.

Core's PackedLayout writes t = text_len for a first-frame keyframe and
t = text_len + sum(spans) - FRAME_RESCALE for a last-frame one. Because the
spans of a full clip sum to FRAME_RESCALE * frame_count, both are the same
formula evaluated at p = 0 and p = frame_count - 1.
"""

from comfy.ldm.minimax.model import FRAME_PER_TOKEN

FPS = 24
AUDIO_LATENT_FPS = 40
CLIP_LENGTH = 17
TOKENS_PER_CLIP = 5


def align_frame_count(n):
    while n % CLIP_LENGTH != TOKENS_PER_CLIP:
        n += 1
    return n


def video_latent_t(frame_count):
    if frame_count <= 5:
        return 2
    return ((frame_count - 5) // CLIP_LENGTH) * TOKENS_PER_CLIP + 2


def temporal_shape(length):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def snap_condition_frames(n):
    """Largest count <= n the video VAE encodes without silently dropping frames.

    encode() pads the clip up to a multiple of 17 and then drops the last 3
    latent tokens; that only removes padding when the count is 17k+5. A single
    frame takes the encoder's still-image path and yields exactly one token.
    """
    if n < 5:
        return 1
    while n % CLIP_LENGTH != TOKENS_PER_CLIP:
        n -= 1
    return n


def token_frame_offsets(latent_t):
    """Pixel-frame index each latent token of a chunk starts at."""
    offsets = []
    frame = 0
    for k in range(latent_t):
        offsets.append(frame)
        frame += FRAME_PER_TOKEN[k % TOKENS_PER_CLIP]
    return offsets


def token_slot_for_frame(pixel_index, latent_t):
    """Latent-token slot in a full clip whose pixel-frame start equals pixel_index.

    Every keyframe is recorded with the pixel-frame index its latent token begins
    at (see _run_keyframes). The full-clip packing puts one latent token per entry
    of token_frame_offsets(latent_t); this returns the slot whose offset matches,
    so a caller can write the keyframe latent straight into video[:, :, slot].
    """
    offsets = token_frame_offsets(latent_t)
    for s, o in enumerate(offsets):
        if o == pixel_index:
            return s
    raise ValueError("MiniMax H3: pixel frame %d is not a latent-token boundary for "
                     "latent_t=%d (offsets start %s...)" % (pixel_index, latent_t, offsets[:4]))


def build_pin_spec(keyframes, latent_t):
    """List of (token_slot, keyframe_latent[1,24,1,h,w]) for every keyframe."""
    return [(token_slot_for_frame(kf["resolved_frame_index"], latent_t), kf["latent"])
            for kf in keyframes]

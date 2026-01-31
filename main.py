import argparse
import librosa
import numpy
from PIL import Image
import colorsys
import math
import hashlib


def main():
    parser=argparse.ArgumentParser(prog='synical',description='Create images from audio')
    parser.add_argument("input",help='Path to audio file')
    parser.add_argument("-o", "--output", default="wallpaper.png", help="Path to save the generated image")
    parser.add_argument("-s", "--size", type=int, default=1080, help="Vertical resolution of the wallpaper")
    parser.add_argument("-t", "--type", choices=["gradient"], default="gradient", help="Type of wallpaper pattern to generate")
    parser.add_argument("--vibrancy", type=float, default=0, help="Color vibrancy (0.0 = muted, 1.0 = default)")
    parser.add_argument("--gradient", choices=["linear", "radial"], default="linear", help="Gradient style")
    args=parser.parse_args()
    print('Loading file')
    audio,sr=librosa.load(args.input,duration=30)
    print('Loading mfcc')
    mfcc=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=40)
    norm=((mfcc-numpy.min(mfcc))/(numpy.max(mfcc)-numpy.min(mfcc)))*255
    
    img = create_gradient_wallpaper(norm, args.size, args.vibrancy, 1920, args.gradient)
    
    # Crop to remove red band from left edge
    crop_box = (10, 0, 1920, args.size)
    img = img.crop(crop_box)
    
    img.save(args.output)
    print(f'Wallpaper saved as {args.output}')


        
        
    






def create_gradient_wallpaper(norm, size, vibrancy, out_width, gradient):
    """Create vibrant gradient wallpaper with smooth hue variations"""
    src_width = int(norm.shape[1])
    width = int(out_width)
    img = Image.new('RGB', (width, size))
    pixels = img.load()

    vibrancy = float(vibrancy)
    if vibrancy < 0.0:
        vibrancy = 0.0
    if vibrancy > 2.0:
        vibrancy = 2.0

    if src_width <= 1:
        return img

    # Deterministic 5-color palette seeded from audio
    seed_hash = hashlib.sha256(norm.tobytes()).hexdigest()
    seed_int = int(seed_hash[:16], 16)
    hues = []
    for i in range(5):
        hue = ((seed_int * (i + 1) * 1103515245 + 12345) % (1 << 31)) / (1 << 31)
        hues.append(hue)
    hues.sort()
    
    # Ensure we have at least 3 distinct colors by spacing them out
    if len(hues) >= 3:
        # If colors are too close, redistribute them
        min_spacing = 0.12  # minimum distance between colors
        for i in range(1, len(hues)):
            if hues[i] - hues[i-1] < min_spacing:
                hues[i] = hues[i-1] + min_spacing
        # Normalize back to [0,1]
        if hues[-1] > 1.0:
            scale = 1.0 / hues[-1]
            hues = [h * scale for h in hues]

    def get_smooth_hue(pos):
        # pos in [0,1] across the gradient
        segment = pos * 4.0
        idx = int(segment)
        t = segment - idx
        if idx >= 4:
            return hues[4]
        h0 = hues[idx]
        h1 = hues[idx + 1]
        # Wrap shortest path around circle
        diff = (h1 - h0)
        if diff > 0.5:
            diff -= 1.0
        elif diff < -0.5:
            diff += 1.0
        result = (h0 + diff * t) % 1.0
        # Map to [0.1, 1.1] range to avoid red boundary
        return (result + 0.1) % 1.0

    if gradient == "radial":
        return create_radial_gradient_wallpaper(norm, size, vibrancy, out_width, hues, get_smooth_hue)

    # Smooth hue tracking (avoids sudden hue jumps but keeps full spectrum)
    smooth_alpha = 0.08
    hue_cos = 1.0
    hue_sin = 0.0
    sat_smooth = 0.85

    for x in range(width):
        # Normalized audio features (0..1)
        t = (x * (src_width - 1)) / max(1, (width - 1))
        i0 = int(t)
        i1 = i0 + 1
        if i1 >= src_width:
            i1 = src_width - 1
        frac = t - i0

        def lerp_row(row_idx):
            v0 = float(norm[row_idx, i0])
            v1 = float(norm[row_idx, i1])
            return (v0 * (1.0 - frac) + v1 * frac) / 255.0

        a1 = lerp_row(5)
        a2 = lerp_row(15)
        a3 = lerp_row(25)
        a4 = lerp_row(35)

        # Base hue from 5-color palette + audio modulation
        base_h = get_smooth_hue(x / max(1, (width - 1)))
        hue_mod = ((a2 - 0.5) * 0.22 + (a3 - 0.5) * 0.10) * vibrancy
        hue_raw = (base_h + hue_mod) % 1.0

        # Circular smoothing for hue
        theta = hue_raw * math.tau
        hue_cos = (1.0 - smooth_alpha) * hue_cos + smooth_alpha * math.cos(theta)
        hue_sin = (1.0 - smooth_alpha) * hue_sin + smooth_alpha * math.sin(theta)
        hue = (math.atan2(hue_sin, hue_cos) / math.tau) % 1.0

        # Saturation: keep it high, but still audio-reactive and smooth
        sat_floor = 0.45 + 0.20 * vibrancy
        sat_span = 0.25 + 0.30 * vibrancy
        sat_raw = sat_floor + sat_span * (0.35 * a4 + 0.65 * a1)
        sat_smooth = (1.0 - smooth_alpha) * sat_smooth + smooth_alpha * sat_raw
        sat = max(0.40, min(0.98, sat_smooth))

        # Value/brightness varies vertically + audio energy, clamped to avoid whiteouts
        energy = 0.55 * a1 + 0.45 * a3

        for y in range(size):
            vpos = y / max(1, (size - 1))
            v_raw = 0.22 + 0.70 * energy
            v_raw *= (0.92 - 0.45 * (vpos ** 1.15))

            # subtle shimmer (kept small so it doesn't clip to white)
            shimmer = 0.0
            val = max(0.06, min(0.92, v_raw + shimmer))

            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            pixels[x, y] = (int(r * 255), int(g * 255), int(b * 255))
    
    return img


def create_radial_gradient_wallpaper(norm, size, vibrancy, out_width, hues, get_smooth_hue):
    src_width = int(norm.shape[1])
    width = int(out_width)
    img = Image.new('RGB', (width, size))
    pixels = img.load()

    vibrancy = float(vibrancy)
    if vibrancy < 0.0:
        vibrancy = 0.0
    if vibrancy > 2.0:
        vibrancy = 2.0

    if src_width <= 1:
        return img

    cx = (width - 1) * 0.5
    cy = (size - 1) * 0.5
    max_r = math.hypot(cx, cy)
    if max_r <= 0.0:
        return img

    # Precompute smooth HSV + brightness per radius (avoids artifacts)
    steps = int(max_r) + 2
    smooth_alpha = 0.08
    hue_cos = 1.0
    hue_sin = 0.0
    sat_smooth = 0.85

    hue_by_r = [0.0] * steps
    sat_by_r = [0.0] * steps
    val_by_r = [0.0] * steps

    for ri in range(steps):
        rnorm = ri / (steps - 1)

        # Map radius to MFCC timeline position
        t = rnorm * (src_width - 1)
        i0 = int(t)
        i1 = i0 + 1
        if i1 >= src_width:
            i1 = src_width - 1
        frac = t - i0

        def lerp_row(row_idx):
            v0 = float(norm[row_idx, i0])
            v1 = float(norm[row_idx, i1])
            return (v0 * (1.0 - frac) + v1 * frac) / 255.0

        a1 = lerp_row(5)
        a2 = lerp_row(15)
        a3 = lerp_row(25)
        a4 = lerp_row(35)

        base_h = get_smooth_hue(rnorm)
        hue_mod = ((a2 - 0.5) * 0.22 + (a3 - 0.5) * 0.10) * vibrancy
        hue_raw = (base_h + hue_mod) % 1.0

        theta = hue_raw * math.tau
        hue_cos = (1.0 - smooth_alpha) * hue_cos + smooth_alpha * math.cos(theta)
        hue_sin = (1.0 - smooth_alpha) * hue_sin + smooth_alpha * math.sin(theta)
        hue = (math.atan2(hue_sin, hue_cos) / math.tau) % 1.0

        sat_floor = 0.45 + 0.20 * vibrancy
        sat_span = 0.25 + 0.30 * vibrancy
        sat_raw = sat_floor + sat_span * (0.35 * a4 + 0.65 * a1)
        sat_smooth = (1.0 - smooth_alpha) * sat_smooth + smooth_alpha * sat_raw
        sat = max(0.40, min(0.98, sat_smooth))

        energy = 0.55 * a1 + 0.45 * a3

        # Radial brightness: bright in the center, darker towards edges
        v_raw = 0.25 + 0.70 * energy
        v_raw *= (0.96 - 0.60 * (rnorm ** 1.35))
        val = max(0.06, min(0.92, v_raw))

        hue_by_r[ri] = hue
        sat_by_r[ri] = sat
        val_by_r[ri] = val

    for y in range(size):
        dy = y - cy
        for x in range(width):
            dx = x - cx
            r = math.hypot(dx, dy)
            rnorm = r / max_r
            if rnorm > 1.0:
                rnorm = 1.0
            ri = int(rnorm * (steps - 1))
            hue = hue_by_r[ri]
            sat = sat_by_r[ri]
            val = val_by_r[ri]
            rr, gg, bb = colorsys.hsv_to_rgb(hue, sat, val)
            pixels[x, y] = (int(rr * 255), int(gg * 255), int(bb * 255))

    return img

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Preprocess physical proxy indicators from street view images.

This script computes 5 physical proxies (as per paper Section 3):
1. NDVI proxy (Vegetation): (G - R) / (G + R)
2. Tree canopy proxy: Green pixel fraction using HSV mask
3. Impervious surface proxy: Gray/white pixel fraction
4. Albedo proxy (Reflectance): Overall brightness
5. Shadow fraction proxy: Dark pixel fraction

from street view images for each tract.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_ndvi_proxy(image: Image.Image) -> float:
    """
    Compute NDVI proxy from RGB image.
    NDVI ≈ (G - R) / (G + R) for RGB images
    """
    img_array = np.array(image.convert('RGB'))
    r = img_array[:, :, 0].astype(np.float32)
    g = img_array[:, :, 1].astype(np.float32)
    
    numerator = g - r
    denominator = g + r + 1e-6
    ndvi = numerator / denominator
    
    return float(np.mean(ndvi))


def compute_tree_canopy_proxy(image: Image.Image) -> float:
    """
    Compute tree canopy proxy (Section 3.4, ii).
    Combines "green-dominant" mask {g > 100, g > r, g > b} 
    with HSV-based green mask, then averages the two pixel fractions.
    """
    img_array = np.array(image.convert('RGB'))
    r = img_array[:, :, 0].astype(np.float32)
    g = img_array[:, :, 1].astype(np.float32)
    b = img_array[:, :, 2].astype(np.float32)
    
    # Method 1: Green-dominant mask {g > 100, g > r, g > b}
    green_dominant_mask = (g > 100) & (g > r) & (g > b)
    green_dominant_fraction = np.mean(green_dominant_mask.astype(np.float32))
    
    # Method 2: HSV-based green mask
    # Convert RGB to HSV
    img_hsv = image.convert('HSV')
    hsv_array = np.array(img_hsv)
    h = hsv_array[:, :, 0].astype(np.float32)  # Hue: 0-179
    s = hsv_array[:, :, 1].astype(np.float32)  # Saturation: 0-255
    v = hsv_array[:, :, 2].astype(np.float32)  # Value: 0-255
    
    # Green hue range: typically 60-120 degrees (in 0-179 scale: 30-60)
    # Sufficient saturation (> 50) and brightness (> 50)
    hsv_green_mask = (h >= 30) & (h <= 60) & (s > 50) & (v > 50)
    hsv_green_fraction = np.mean(hsv_green_mask.astype(np.float32))
    
    # Average of the two methods, clipped to [0, 1]
    canopy_fraction = (green_dominant_fraction + hsv_green_fraction) / 2.0
    canopy_fraction = np.clip(canopy_fraction, 0.0, 1.0)
    
    return float(canopy_fraction)


def compute_impervious_proxy(image: Image.Image) -> float:
    """
    Compute impervious surface proxy (Section 3.4, iii).
    Detects bright, low-saturation pixels {G > 150, S < 0.2} 
    and blends with NDBI-like term (r-g)/(r+g+10^-6) using weights 0.6 and 0.4.
    """
    img_array = np.array(image.convert('RGB'))
    r = img_array[:, :, 0].astype(np.float32)
    g = img_array[:, :, 1].astype(np.float32)
    b = img_array[:, :, 2].astype(np.float32)
    
    # Compute luminance G = 0.299*r + 0.587*g + 0.114*b
    G = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Convert to HSV for saturation
    img_hsv = image.convert('HSV')
    hsv_array = np.array(img_hsv)
    s_hsv = hsv_array[:, :, 1].astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    # Method 1: Bright, low-saturation mask {G > 150, S < 0.2}
    bright_low_sat_mask = (G > 150) & (s_hsv < 0.2)
    bright_low_sat_fraction = np.mean(bright_low_sat_mask.astype(np.float32))
    
    # Method 2: NDBI-like term (r - g) / (r + g + 10^-6)
    numerator = r - g
    denominator = r + g + 1e-6
    ndbi_like = numerator / denominator
    # Normalize to [0, 1] by taking positive values and scaling
    ndbi_like_normalized = np.clip((ndbi_like + 1.0) / 2.0, 0.0, 1.0)
    ndbi_fraction = np.mean(ndbi_like_normalized)
    
    # Blend with weights 0.6 and 0.4, then clip to [0, 1]
    impervious_fraction = 0.6 * bright_low_sat_fraction + 0.4 * ndbi_fraction
    impervious_fraction = np.clip(impervious_fraction, 0.0, 1.0)
    
    return float(impervious_fraction)


def compute_albedo_proxy(image: Image.Image) -> float:
    """
    Compute albedo proxy (Section 3.4, iv).
    Uses mean luminance G/255, clipped to [0.1, 0.9].
    High albedo (white roof) → Day LST ↓
    """
    img_array = np.array(image.convert('RGB'))
    r = img_array[:, :, 0].astype(np.float32)
    g = img_array[:, :, 1].astype(np.float32)
    b = img_array[:, :, 2].astype(np.float32)
    
    # Compute luminance G = 0.299*r + 0.587*g + 0.114*b
    G = 0.299 * r + 0.587 * g + 0.114 * b
    mean_luminance = np.mean(G)
    
    # Normalize to [0, 1] and clip to [0.1, 0.9] as per paper
    albedo = mean_luminance / 255.0
    albedo = np.clip(albedo, 0.1, 0.9)
    
    return float(albedo)


def compute_shadow_fraction_proxy(image: Image.Image) -> float:
    """
    Compute shadow fraction proxy (Section 3.4, v).
    Takes fraction of very dark pixels {G < 50}, clipped to [0, 0.5].
    Shadow areas typically have lower temperature.
    """
    img_array = np.array(image.convert('RGB'))
    r = img_array[:, :, 0].astype(np.float32)
    g = img_array[:, :, 1].astype(np.float32)
    b = img_array[:, :, 2].astype(np.float32)
    
    # Compute luminance G = 0.299*r + 0.587*g + 0.114*b
    G = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Shadow detection: very dark pixels {G < 50}
    shadow_mask = G < 50
    shadow_fraction = np.mean(shadow_mask.astype(np.float32))
    
    # Clip to [0, 0.5] as per paper
    shadow_fraction = np.clip(shadow_fraction, 0.0, 0.5)
    
    return float(shadow_fraction)


def process_tract(tract_dir: Path, output_dir: Path):
    """Process all street view images for a tract and compute proxies"""
    if not tract_dir.exists():
        return None
    
    image_files = sorted([
        f for f in tract_dir.iterdir()
        if f.suffix.lower() in {'.jpg', '.png', '.jpeg'}
    ])
    
    if not image_files:
        return None
    
    proxies = {
        'ndvi_proxy': [],
        'tree_canopy_proxy': [],
        'impervious_proxy': [],
        'albedo_proxy': [],
        'shadow_fraction': [],  # 5th proxy as per paper
    }
    
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB')
            proxies['ndvi_proxy'].append(compute_ndvi_proxy(img))
            proxies['tree_canopy_proxy'].append(compute_tree_canopy_proxy(img))
            proxies['impervious_proxy'].append(compute_impervious_proxy(img))
            proxies['albedo_proxy'].append(compute_albedo_proxy(img))
            proxies['shadow_fraction'].append(compute_shadow_fraction_proxy(img))
        except Exception as e:
            print(f"Warning: Failed to process {img_path}: {e}")
            continue
    
    if not proxies['ndvi_proxy']:
        return None
    
    # Average across all images
    tract_id = tract_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for proxy_name, values in proxies.items():
        mean_value = np.mean(values)
        output_path = output_dir / f'{tract_id}_{proxy_name}.npy'
        np.save(output_path, mean_value)
    
    return {name: np.mean(values) for name, values in proxies.items()}


def main():
    parser = argparse.ArgumentParser(description='Preprocess physical proxy indicators')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Root data directory')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Data split to process')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: data_dir/concepts/split)')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    street_dir = data_dir / 'street' / args.split
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / 'concepts' / args.split
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tract_dirs = sorted([d for d in street_dir.iterdir() if d.is_dir()])
    
    print(f"Processing {len(tract_dirs)} tracts...")
    results = {}
    for tract_dir in tqdm(tract_dirs):
        proxies = process_tract(tract_dir, output_dir)
        if proxies:
            results[tract_dir.name] = proxies
    
    print(f"✅ Processed {len(results)} tracts")
    print(f"✅ Proxy files saved to {output_dir}")


if __name__ == '__main__':
    main()


"""
Dataset for loading Satellite, Street View, Mobility, and Proxy data

Handles multi-modal data alignment:
- Satellite Image: Single RGB image per tract
- Street View: Bag of images (K images per tract), returns stacked tensor [B, K, C, H, W]
- Mobility Vector: [B, 168, 3] weekly patterns (24×7 hours, 3 features)
- Proxy Indicators: 5 physical proxies (NDVI, tree canopy, impervious, albedo, shadow)
"""
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AESPADataset(Dataset):
    """Dataset for AESPA with Sat, SV, Mobility, and Proxies"""

    DEFAULT_CONCEPTS = [
        "ndvi_proxy",
        "tree_canopy_proxy",
        "impervious_proxy",
        "albedo_proxy",
        "shadow_fraction",  # 5th proxy as per paper
    ]

    def __init__(self, data_dir, split='train', config=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.config = config or {}

        self.street_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.satellite_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Load concept proxies
        self.concept_names = self.config.get(
            'concept_names', self.DEFAULT_CONCEPTS
        )
        self.concept_dir = self.data_dir / 'concepts' / self.split

        # Load normalization parameters
        self.lst_mean, self.lst_std = self._load_normalization_params()
        self.normalize_lst = self.config.get('normalize_lst', True)

        self.data_list = self._load_data_list()
        self.num_street_images = self.config.get('num_images', 16)

    def _load_normalization_params(self):
        """Load normalization parameters from data directory"""
        norm_path = self.data_dir / 'normalization.json'
        if norm_path.exists():
            with norm_path.open() as f:
                norm_data = json.load(f)
                return norm_data.get('mean', 25.0), norm_data.get('std', 5.0)
        return 25.0, 5.0  # Default values

    def _load_data_list(self):
        """Load list of data samples"""
        data_list = []
        lst_dir = self.data_dir / 'lst' / self.split
        street_dir = self.data_dir / 'street' / self.split
        satellite_dir = self.data_dir / 'satellite' / self.split
        mobility_dir = self.data_dir / 'mobility' / self.split

        if not lst_dir.exists():
            raise FileNotFoundError(
                f'LST 目录 {lst_dir} 不存在，请先运行数据预处理脚本。')

        for path in sorted(lst_dir.iterdir()):
            if path.suffix.lower() not in {'.npy', '.npz'}:
                continue
            tract_id = path.stem
            data_list.append({
                'tract_id': tract_id,
                'lst_path': str(path),
                'street_dir': str(street_dir / tract_id),
                'satellite_path': str(satellite_dir / f'{tract_id}.png'),
                'mobility_path': str(mobility_dir / f'{tract_id}.npy')
                if mobility_dir.exists() else None
            })
        return data_list

    def _load_street_images(self, street_dir):
        """Load and transform street view images (Bag of images)"""
        street_dir = Path(street_dir)
        if not street_dir.exists():
            return [torch.randn(3, 224, 224) for _ in range(self.num_street_images)]

        image_files = sorted([
            f for f in street_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.png', '.jpeg'}
        ])
        if len(image_files) >= self.num_street_images:
            selected = np.random.choice(
                image_files, self.num_street_images, replace=False)
        else:
            selected = list(image_files)
            while len(selected) < self.num_street_images and image_files:
                selected.append(np.random.choice(image_files))

        images = []
        for img_path in selected:
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(self.street_transform(img))
            except Exception:
                images.append(torch.randn(3, 224, 224))

        if not images:
            images = [torch.randn(3, 224, 224) for _ in range(self.num_street_images)]

        return images

    def _load_satellite_image(self, satellite_path):
        """Load and transform satellite image"""
        satellite_path = Path(satellite_path)
        if not satellite_path.exists():
            return torch.randn(3, 256, 256)
        try:
            img = Image.open(satellite_path).convert('RGB')
            return self.satellite_transform(img)
        except Exception:
            return torch.randn(3, 256, 256)

    def _load_lst(self, lst_path):
        """Load LST data (day temperature)"""
        try:
            data = np.load(lst_path)
            if isinstance(data, np.ndarray):
                day_temp = float(data[0]) if len(data) > 0 else 25.0
            else:
                day_temp = float(data.get('day', 25.0))
            
            # Normalize if enabled
            if self.normalize_lst:
                day_temp = (day_temp - self.lst_mean) / self.lst_std
            
            return np.array([day_temp], dtype=np.float32)
        except Exception:
            default_day = 25.0
            if self.normalize_lst:
                default_day = (default_day - self.lst_mean) / self.lst_std
            return np.array([default_day], dtype=np.float32)

    def _load_mobility(self, mobility_path):
        """Load mobility data [168, 3] for 24×7 hours"""
        if mobility_path is None or not os.path.exists(mobility_path):
            return np.zeros((168, 3), dtype=np.float32)
        try:
            return np.load(mobility_path).astype(np.float32)
        except Exception:
            return np.zeros((168, 3), dtype=np.float32)

    def _load_concepts(self, tract_id):
        """Load concept proxy values (5 physical proxies)"""
        concepts = {}
        for concept_name in self.concept_names:
            concept_path = self.concept_dir / f'{tract_id}_{concept_name}.npy'
            if concept_path.exists():
                concepts[concept_name] = float(np.load(concept_path))
            else:
                concepts[concept_name] = 0.0
        return [concepts.get(name, 0.0) for name in self.concept_names]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Returns:
            street_images: [K, C, H, W] - stacked street view images (Bag of images)
            satellite_image: [C, H, W] - single satellite image
            lst_labels: [1] or [2] - day temperature (and optionally night)
            mobility_data: [168, 3] - weekly mobility patterns
            concept_targets: [5] - 5 physical proxy values
            conditional_tokens: dict - tokens for FiLM conditioning
        """
        item = self.data_list[idx]
        tract_id = item['tract_id']

        # Street View: Bag of images, stacked to [K, C, H, W]
        street_images = torch.stack(self._load_street_images(item['street_dir']))
        satellite_image = self._load_satellite_image(item['satellite_path'])
        lst = torch.tensor(self._load_lst(item['lst_path']), dtype=torch.float32)
        mobility = torch.tensor(self._load_mobility(item['mobility_path']), dtype=torch.float32)
        concept_targets = torch.tensor(
            self._load_concepts(tract_id),
            dtype=torch.float32
        )

        # Conditional tokens (city_id, day_of_week, etc.)
        tokens = {
            "city_id": torch.tensor([0.0], dtype=torch.float32),  # Placeholder
        }

        return {
            "street_images": street_images,  # [K, C, H, W] - Bag of images
            "satellite_image": satellite_image,  # [C, H, W]
            "lst_labels": lst,  # [1] or [2]
            "mobility_data": mobility,  # [168, 3]
            "conditional_tokens": tokens,
            "concept_targets": concept_targets,  # [5] - 5 physical proxies
            "tract_id": tract_id,
        }

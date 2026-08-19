#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Ünsal Öztürk <unsal.ozturk@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Script: recordio.py
# Standalone reader for the MXNet RecordIO format; reimplemented here, no MXNet dependency.
#
"""
RecordIO DataManager - Standalone Implementation

Efficient random access to images stored in MXNet RecordIO binary format
without requiring MXNet as a dependency.

Dependencies: numpy, Pillow (PIL), optional: PyTorch (for to_tensor)

RecordIO Format:
    - Index file (.idx): record_id<TAB>byte_offset mapping
    - Record file (.rec): binary records with header + image data
    - Access is position-based: dm[i] returns i-th record in index order

Usage:
    from recordio_standalone import RecordIODataManager
    
    dm = RecordIODataManager('train.idx', 'train.rec')
    img = dm[0]  # PIL Image
    batch = dm[[0, 1, 2]]  # List of images
    
    dm = RecordIODataManager('train.idx', 'train.rec', to_tensor=True, collate=True)
    batch = dm[0:10]  # Tensor of shape [10, 3, H, W]

Author: Unsal Ozturk
License: MIT
"""

from typing import Union, Optional, Any
from collections import namedtuple
from pathlib import Path
import numpy as np
import struct
import io
from PIL import Image


class RecordIODataManager:
    """
    Read images from MXNet RecordIO format with position-based indexing.
    
    Args:
        idx_path: Path to .idx index file
        rec_path: Path to .rec binary record file
        transform: Optional transform to apply to PIL Images
        to_tensor: Convert images to PyTorch tensors [C,H,W] in range [0,1]
        collate: Stack batch results into single tensor/array
    
    Examples:
        dm = RecordIODataManager('train.idx', 'train.rec')
        img = dm[0]  # PIL Image
        batch = dm[[0,1,2]]  # List of PIL Images
        
        dm = RecordIODataManager('train.idx', 'train.rec', to_tensor=True, collate=True)
        tensor = dm[0:10]  # Tensor [10, 3, H, W]
    """
    
    _IRHeader = namedtuple('IRHeader', ['flag', 'label', 'id', 'id2'])
    _IR_FORMAT = 'IfQQ'
    _IR_SIZE = struct.calcsize(_IR_FORMAT)
    
    class _Reader:
        """Internal RecordIO file reader for low-level binary access."""
        
        def __init__(self, idx_path: Union[str, Path], rec_path: Union[str, Path]):
            self.idx_path = Path(idx_path)
            self.rec_path = Path(rec_path)
            self.idx = {}
            self.keys = []
            self.rec_file = open(rec_path, 'rb')
            self._load_idx()
        
        def _load_idx(self):
            """Load index file: build record_id -> byte_offset mapping and keys list."""
            with open(self.idx_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        key = int(parts[0])
                        offset = int(parts[1])
                        self.idx[key] = offset
                        self.keys.append(key)
        
        def read_idx(self, record_key: int) -> bytes:
            """
            Read a record by its ID from the binary file.
            
            Args:
                record_key: Record ID from index file
            
            Returns:
                Raw bytes containing IRHeader + image data
            """
            if record_key not in self.idx:
                raise KeyError(f"Record key {record_key} not found in index")
            
            offset = self.idx[record_key]
            self.rec_file.seek(offset)
            
            magic = struct.unpack('<I', self.rec_file.read(4))[0]
            cflag_length = struct.unpack('<I', self.rec_file.read(4))[0]
            length = cflag_length & ((1 << 29) - 1)
            
            return self.rec_file.read(length)
        
        def __len__(self):
            return len(self.keys)
        
        def close(self):
            if hasattr(self, 'rec_file') and self.rec_file:
                self.rec_file.close()
        
        def __del__(self):
            self.close()
    
    def __init__(self,
                 idx_path: Union[str, Path],
                 rec_path: Union[str, Path],
                 transform: Optional[Any] = None,
                 to_tensor: bool = False,
                 collate: bool = False):
        self.reader = self._Reader(idx_path, rec_path)
        self.transform = transform
        self.to_tensor = to_tensor
        self.collate = collate
    
    @classmethod
    def _unpack_recordio(cls, data: bytes):
        """
        Unpack RecordIO data into header and image bytes.
        
        Args:
            data: Raw record bytes
        
        Returns:
            (header, image_bytes) where header is IRHeader namedtuple
        """
        header = cls._IRHeader(*struct.unpack(cls._IR_FORMAT, data[:cls._IR_SIZE]))
        remaining = data[cls._IR_SIZE:]
        
        if header.flag > 0:
            label_array = np.frombuffer(remaining[:header.flag*4], np.float32, header.flag)
            header = header._replace(label=label_array)
            remaining = remaining[header.flag*4:]
        
        return header, remaining
    
    def get_raw_bytes(self, position: int) -> bytes:
        """Get raw record bytes at position without decoding image."""
        record_key = self.reader.keys[position]
        return self.reader.read_idx(record_key)
    
    def get_raw_bytes_batch(self, positions: Union[list, np.ndarray, slice]) -> list:
        """Get raw bytes for multiple positions."""
        if isinstance(positions, slice):
            indices = list(range(*positions.indices(len(self))))
        elif isinstance(positions, np.ndarray):
            if positions.dtype == bool:
                indices = np.where(positions)[0].tolist()
            else:
                indices = positions.tolist()
        else:
            indices = list(positions)
        
        return [self.get_raw_bytes(int(idx)) for idx in indices]
    
    def _fetch_item(self, position: int):
        """Fetch and decode image at position."""
        data = self.get_raw_bytes(position)
        _, img_bytes = self._unpack_recordio(data)
        return Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    def __getitem__(self, key: Union[int, list, np.ndarray, slice]) -> Any:
        """
        Get image(s) by position(s).
        
        Args:
            key: Position index (int), list/array of indices, or slice
        
        Returns:
            Single image, list of images, or stacked tensor/array
        """
        if isinstance(key, (int, np.integer)):
            img = self._fetch_item(int(key))
            
            if self.transform:
                img = self.transform(img)
            
            if self.to_tensor:
                import torch
                img_array = np.array(img).astype(np.float32) / 255.0
                img = torch.from_numpy(img_array.transpose(2, 0, 1))
            
            return img
        
        if isinstance(key, slice):
            indices = list(range(*key.indices(len(self))))
        elif isinstance(key, np.ndarray):
            indices = np.where(key)[0].tolist() if key.dtype == bool else key.tolist()
        else:
            indices = list(key)
        
        items = [self[int(idx)] for idx in indices]
        
        if not self.collate or not items:
            return items
        
        if self.to_tensor:
            import torch
            return torch.stack(items)
        return np.stack(items)
    
    def __len__(self) -> int:
        return len(self.reader)
    
    def close(self):
        self.reader.close()
    
    def __del__(self):
        self.close()
#!/usr/bin/env python3
"""
Objaverse Dataset Utilities
Handles downloading and managing 3D models from Objaverse dataset on HuggingFace
"""

import os
import logging
import tempfile
import urllib.request
import urllib.error
import numpy as np

logger = logging.getLogger(__name__)

# HuggingFace Objaverse dataset base URL
OBJAVERSE_BASE_URL = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs"

class ObjaverseManager:
    """
    Manager for loading 3D models from Objaverse dataset
    """
    def __init__(self, csv_path, temp_dir=None, max_retries=3, selection=None):
        """
        Args:
            csv_path: Path to CSV file containing objaverse IDs
            temp_dir: Directory for temporary downloads (if None, uses system temp)
            max_retries: Maximum number of download retries
            selection: Optional list/set of specific object IDs to use (filters the CSV)
        """
        self.csv_path = csv_path
        self.max_retries = max_retries
        self.selection = set(selection) if selection is not None else None
        
        # Create temp directory for downloads
        if temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="objaverse_")
            self._own_temp_dir = True
        else:
            self.temp_dir = temp_dir
            os.makedirs(self.temp_dir, exist_ok=True)
            self._own_temp_dir = False
            
        logger.info(f"Objaverse temp directory: {self.temp_dir}")
        
        # Load the CSV
        self._load_csv()
        
    def _load_csv(self):
        """Load Objaverse IDs from CSV file"""
        self.prefixes = []
        self.object_ids = []
        
        with open(self.csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) != 2:
                    continue
                    
                prefix, obj_id = parts[0].strip(), parts[1].strip()
                
                # Filter by selection if provided
                if self.selection is not None and obj_id not in self.selection:
                    continue
                    
                self.prefixes.append(prefix)
                self.object_ids.append(obj_id)
        
        self.num_objects = len(self.object_ids)
        logger.info(f"Loaded {self.num_objects} Objaverse object IDs from {self.csv_path}")
        
        if self.num_objects == 0:
            raise ValueError(f"No valid Objaverse IDs found in {self.csv_path}")
    
    def get_download_url(self, prefix, object_id):
        """
        Construct download URL for an Objaverse object
        
        Args:
            prefix: Prefix from CSV (e.g., "000-001")
            object_id: Object ID hash
            
        Returns:
            Download URL string
        """
        return f"{OBJAVERSE_BASE_URL}/{prefix}/{object_id}.glb"
    
    def download_object(self, prefix, object_id, max_retries=None):
        """
        Download a single Objaverse object to temp directory
        
        Args:
            prefix: Prefix from CSV
            object_id: Object ID hash
            max_retries: Override default max_retries
            
        Returns:
            Path to downloaded file, or None if download failed
        """
        if max_retries is None:
            max_retries = self.max_retries
            
        url = self.get_download_url(prefix, object_id)
        temp_path = os.path.join(self.temp_dir, f"{object_id}.glb")
        
        # Try to download with retries
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {object_id} from {url} (attempt {attempt + 1}/{max_retries})")
                
                # Download with timeout
                with urllib.request.urlopen(url, timeout=30) as response:
                    with open(temp_path, 'wb') as f:
                        f.write(response.read())
                
                # Verify file exists and has content
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    logger.info(f"Successfully downloaded {object_id} ({os.path.getsize(temp_path)} bytes)")
                    return temp_path
                else:
                    logger.warning(f"Downloaded file is empty or missing: {temp_path}")
                    
            except urllib.error.HTTPError as e:
                logger.warning(f"HTTP error {e.code} downloading {object_id}: {e.reason}")
                if e.code == 404:
                    # File not found, no point retrying
                    break
            except urllib.error.URLError as e:
                logger.warning(f"URL error downloading {object_id}: {e.reason}")
            except Exception as e:
                logger.warning(f"Error downloading {object_id}: {e}")
        
        return None
    
    def sample_random(self):
        """
        Sample a random object and download it
        
        Returns:
            dict with keys: 'file_path', 'object_id', 'prefix', or None if download failed
        """
        if self.num_objects == 0:
            return None
            
        # Sample random index
        idx = np.random.randint(0, self.num_objects)
        prefix = self.prefixes[idx]
        object_id = self.object_ids[idx]
        
        # Download the file
        file_path = self.download_object(prefix, object_id)
        
        if file_path is None:
            return None
            
        return {
            'file_path': file_path,
            'object_id': object_id,
            'prefix': prefix,
        }
    
    def cleanup_file(self, file_path):
        """Remove a downloaded file"""
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up {file_path}: {e}")
    
    def cleanup_all(self):
        """Clean up all temporary files"""
        if self._own_temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {self.temp_dir}: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup_all()


def load_objaverse_selection(selection_path):
    """
    Load a selection file containing object IDs to filter
    
    Args:
        selection_path: Path to text file with one object ID per line
        
    Returns:
        List of object IDs
    """
    if selection_path is None or not os.path.exists(selection_path):
        return None
        
    selection = []
    with open(selection_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                selection.append(line)
    
    logger.info(f"Loaded {len(selection)} object IDs from selection file")
    return selection


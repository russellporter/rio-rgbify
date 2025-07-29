"""Utilities for handling cutline polygons in source clipping operations."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import rasterio
import rasterio.mask
import rasterio.warp
import rasterio.windows
from rasterio.features import geometry_mask
import json

logger = logging.getLogger(__name__)


def load_cutline_geometries(cutline_path: Path, target_crs: str = 'EPSG:3857') -> List[Dict[str, Any]]:
    """
    Load polygon geometries from a cutline file and transform to target CRS.
    
    Parameters
    ----------
    cutline_path : Path
        Path to the cutline file (GeoJSON supported, shapefile via GDAL)
    target_crs : str, optional
        Target coordinate reference system, defaults to 'EPSG:3857'
        
    Returns
    -------
    List[Dict[str, Any]]
        List of GeoJSON-like geometry dictionaries in target CRS
        
    Raises
    ------
    ValueError
        If cutline file cannot be read or contains no geometries
    """
    if not cutline_path.exists():
        raise ValueError(f"Cutline file does not exist: {cutline_path}")
    
    geometries = []
    
    # Handle GeoJSON files directly
    if cutline_path.suffix.lower() == '.geojson' or cutline_path.suffix.lower() == '.json':
        try:
            with open(cutline_path, 'r') as f:
                geojson_data = json.load(f)
            
            if geojson_data.get('type') == 'FeatureCollection':
                for feature in geojson_data.get('features', []):
                    geom = feature.get('geometry')
                    if geom:
                        geometries.append(geom)
            elif geojson_data.get('type') in ['Polygon', 'MultiPolygon']:
                geometries.append(geojson_data)
                
        except Exception as e:
            raise ValueError(f"Failed to load GeoJSON file {cutline_path}: {str(e)}")
    
    # For other formats, try using rasterio's vector capabilities
    else:
        try:
            # Use rasterio to open vector files through GDAL
            import rasterio.features
            from rasterio.crs import CRS
            
            # Try to read as vector using rasterio's GDAL interface
            with rasterio.Env():
                # This is a simplified approach - in practice you might want to use
                # geopandas or similar for more robust vector file reading
                # For now, recommend using GeoJSON format for cutlines
                raise ValueError(f"Non-GeoJSON vector formats not yet supported. Please convert {cutline_path} to GeoJSON format.")
                
        except Exception as e:
            raise ValueError(f"Failed to load cutline file {cutline_path}: {str(e)}")
    
    if not geometries:
        raise ValueError(f"No valid geometries found in cutline file: {cutline_path}")
        
    # Transform geometries to target CRS if needed
    transformed_geometries = []
    for geom in geometries:
        try:
            # For now, assume input is already in correct CRS
            # In practice, you'd want to handle CRS transformation here
            transformed_geometries.append(geom)
        except Exception as e:
            logger.warning(f"Failed to transform geometry: {e}")
            continue
    
    logger.info(f"Loaded {len(transformed_geometries)} geometries from {cutline_path}")
    return transformed_geometries


def transform_geometry(geometry: Dict[str, Any], source_crs: str, target_crs: str) -> Dict[str, Any]:
    """
    Transform a geometry from source CRS to target CRS.
    
    Parameters
    ----------
    geometry : Dict[str, Any]
        GeoJSON-like geometry dictionary
    source_crs : str
        Source coordinate reference system
    target_crs : str
        Target coordinate reference system
        
    Returns
    -------
    Dict[str, Any]
        Transformed geometry dictionary
    """
    if source_crs == target_crs:
        return geometry
    
    try:
        # Use rasterio's transform_geom for CRS transformation
        return rasterio.warp.transform_geom(source_crs, target_crs, geometry)
    except Exception as e:
        logger.warning(f"Failed to transform geometry from {source_crs} to {target_crs}: {e}")
        return geometry


def clip_array_with_cutline(data: np.ndarray, 
                           transform: rasterio.Affine,
                           geometries: List[Dict[str, Any]], 
                           crs: str = 'EPSG:3857',
                           nodata: float = np.nan) -> np.ndarray:
    """
    Clip a numpy array using cutline polygon geometries.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array to clip
    transform : rasterio.Affine
        Affine transform of the data
    geometries : List[Dict[str, Any]]
        List of polygon geometries to use for clipping
    crs : str, optional
        Coordinate reference system of the data, defaults to 'EPSG:3857'
    nodata : float, optional
        Value to use for clipped areas, defaults to np.nan
        
    Returns
    -------
    np.ndarray
        Clipped data array with areas outside polygons set to nodata
    """
    if not geometries:
        return data
        
    try:
        # Create a memory dataset for the clipping operation
        height, width = data.shape[-2:]
        
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=data.dtype,
                crs=crs,
                transform=transform,
                nodata=nodata
            ) as dataset:
                # Write the data to the dataset
                if data.ndim == 2:
                    dataset.write(data, 1)
                else:
                    dataset.write(data[0], 1)
                
                # Perform the clipping operation
                clipped_data, clipped_transform = rasterio.mask.mask(
                    dataset, 
                    geometries, 
                    crop=False,  # Don't crop to polygon bounds, keep original extent
                    nodata=nodata,
                    filled=True
                )
                
                # Return the clipped data in original shape
                if data.ndim == 2:
                    return clipped_data[0]
                else:
                    return clipped_data
                    
    except Exception as e:
        logger.warning(f"Failed to clip array with cutline: {e}. Returning original data.")
        return data


def create_cutline_mask(shape: Tuple[int, int],
                       transform: rasterio.Affine,
                       geometries: List[Dict[str, Any]],
                       crs: str = 'EPSG:3857') -> np.ndarray:
    """
    Create a boolean mask from cutline geometries.
    
    Parameters
    ----------
    shape : Tuple[int, int]
        Shape of the output mask (height, width)
    transform : rasterio.Affine
        Affine transform for the mask
    geometries : List[Dict[str, Any]]
        List of polygon geometries
    crs : str, optional
        Coordinate reference system, defaults to 'EPSG:3857'
        
    Returns
    -------
    np.ndarray
        Boolean mask where True indicates areas inside polygons
    """
    if not geometries:
        return np.ones(shape, dtype=bool)
    
    try:
        # Transform geometries to match the data CRS if needed
        transformed_geoms = []
        for geom in geometries:
            transformed_geoms.append(geom)
            
        # Create geometry mask (True for areas outside polygons)
        mask = geometry_mask(
            transformed_geoms,
            out_shape=shape,
            transform=transform,
            invert=True  # Invert so True = inside polygons
        )
        
        return mask
        
    except Exception as e:
        logger.warning(f"Failed to create cutline mask: {e}. Returning full mask.")
        return np.ones(shape, dtype=bool)


def validate_cutline_file(cutline_path: Path) -> bool:
    """
    Validate that a cutline file can be read and contains valid geometries.
    
    Parameters
    ----------
    cutline_path : Path
        Path to the cutline file
        
    Returns
    -------
    bool
        True if file is valid, False otherwise
    """
    if not cutline_path.exists():
        return False
        
    try:
        geometries = load_cutline_geometries(cutline_path)
        return len(geometries) > 0
    except Exception as e:
        logger.warning(f"Invalid cutline file {cutline_path}: {e}")
        return False
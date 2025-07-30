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

try:
    from osgeo import ogr, osr
    HAS_OGR = True
except ImportError:
    HAS_OGR = False

logger = logging.getLogger(__name__)


def load_cutline_geometries(cutline_path: Path, target_crs: str = 'EPSG:3857') -> List[Dict[str, Any]]:
    """
    Load polygon geometries from a cutline file and transform to target CRS.
    
    Supports any vector format readable by GDAL/OGR including:
    - GeoJSON (.geojson, .json)
    - Shapefile (.shp)
    - KML (.kml)
    - GPX (.gpx)
    - And many other GDAL-supported formats
    
    Parameters
    ----------
    cutline_path : Path
        Path to the cutline file
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
    
    # Use OGR if available for all formats
    if HAS_OGR:
        geometries = _load_ogr_geometries(cutline_path, target_crs)
    else:
        raise ValueError(
            f"Cannot read vector files without GDAL/OGR. "
            f"Please install GDAL to use cutline functionality."
        )
    
    if not geometries:
        raise ValueError(f"No valid geometries found in cutline file: {cutline_path}")
    
    logger.info(f"Loaded {len(geometries)} geometries from {cutline_path}")
    return geometries



def load_cutline_ogr_geometries(cutline_path: Path, target_crs: str = 'EPSG:3857') -> List['ogr.Geometry']:
    """
    Load cutline geometries as OGR geometry objects (no JSON conversion).
    
    This is more efficient for repeated spatial operations like clipping.
    """
    if not HAS_OGR:
        raise ValueError("Cannot load OGR geometries without GDAL/OGR installed")
        
    geometries = []
    
    try:
        # Open the vector file with OGR (let OGR auto-detect the driver)
        datasource = ogr.Open(str(cutline_path))
        
        if datasource is None:
            raise ValueError(f"Could not open vector file: {cutline_path}")
        
        # Get the layer (use first layer)
        layer = datasource.GetLayer(0)
        if layer is None:
            raise ValueError(f"No layers found in vector file: {cutline_path}")
        
        # Get source CRS
        source_srs = layer.GetSpatialRef()
        
        # Set up coordinate transformation if needed
        transform = None
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(3857)  # Always target Web Mercator
        
        if source_srs and not source_srs.IsSame(target_srs):
            transform = osr.CoordinateTransformation(source_srs, target_srs)
        
        # Read each feature and extract geometry
        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom is None:
                continue
            
            # Only process polygon geometries
            geom_type = geom.GetGeometryType()
            if geom_type not in [ogr.wkbPolygon, ogr.wkbMultiPolygon]:
                continue
            
            # Clone geometry to avoid issues with datasource cleanup
            cloned_geom = geom.Clone()
            
            # Transform geometry if needed
            if transform:
                cloned_geom.Transform(transform)
            
            # Simplify complex geometries for better performance
            # OGR Simplify takes tolerance as positional argument, not keyword
            simplified_geom = cloned_geom.Simplify(10.0)  # 10 meter tolerance in Web Mercator
            if simplified_geom and not simplified_geom.IsEmpty():
                geometries.append(simplified_geom)
            else:
                geometries.append(cloned_geom)
        
        # Clean up
        datasource = None
        
        logger.info(f"Loaded {len(geometries)} OGR geometries from {cutline_path}")
        
    except Exception as e:
        raise ValueError(f"Failed to load vector file {cutline_path}: {str(e)}")
    
    return geometries


def _load_ogr_geometries(cutline_path: Path, target_crs: str = 'EPSG:3857') -> List[Dict[str, Any]]:
    """Load geometries from any OGR-supported vector format with optimization for large geometries."""
    geometries = []
    
    try:
        # Open the vector file with OGR (let OGR auto-detect the driver)
        datasource = ogr.Open(str(cutline_path))
        
        if datasource is None:
            raise ValueError(f"Could not open vector file: {cutline_path}")
        
        # Get the layer (use first layer)
        layer = datasource.GetLayer(0)
        if layer is None:
            raise ValueError(f"No layers found in vector file: {cutline_path}")
        
        # Get source CRS
        source_srs = layer.GetSpatialRef()
        source_crs = None
        if source_srs:
            source_crs = source_srs.ExportToWkt()
        
        # Set up coordinate transformation if needed
        transform = None
        if source_srs:
            target_srs = osr.SpatialReference()
            if target_crs.startswith('EPSG:'):
                target_srs.ImportFromEPSG(int(target_crs.split(':')[1]))
            else:
                target_srs.ImportFromWkt(target_crs)
            
            if not source_srs.IsSame(target_srs):
                transform = osr.CoordinateTransformation(source_srs, target_srs)
        
        # Read each feature and extract geometry
        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom is None:
                continue
            
            # Only process polygon geometries
            geom_type = geom.GetGeometryType()
            if geom_type not in [ogr.wkbPolygon, ogr.wkbMultiPolygon]:
                continue
            
            # Transform geometry if needed
            if transform:
                geom.Transform(transform)
            
            # Simplify complex geometries for better performance
            # OGR Simplify takes tolerance as positional argument, not keyword
            simplified_geom = geom.Simplify(10.0)  # 10 meter tolerance in Web Mercator
            if simplified_geom and not simplified_geom.IsEmpty():
                geom = simplified_geom
            
            # Convert to GeoJSON-like dictionary
            geom_json_str = geom.ExportToJson()
            if geom_json_str:
                geometry_dict = json.loads(geom_json_str)
                geometries.append(geometry_dict)
        
        # Clean up
        datasource = None
        
        logger.info(f"Loaded and simplified {len(geometries)} geometries from {cutline_path}")
        
    except Exception as e:
        raise ValueError(f"Failed to load vector file {cutline_path}: {str(e)}")
    
    return geometries


def transform_geometry(geometry: Dict[str, Any], source_crs: str, target_crs: str) -> Dict[str, Any]:
    """
    Transform a geometry from source CRS to target CRS.
    
    Parameters
    ----------
    geometry : Dict[str, Any]
        GeoJSON-like geometry dictionary
    source_crs : str
        Source coordinate reference system (EPSG code or WKT)
    target_crs : str
        Target coordinate reference system (EPSG code or WKT)
        
    Returns
    -------
    Dict[str, Any]
        Transformed geometry dictionary
    """
    if source_crs == target_crs:
        return geometry
    
    try:
        # First try using rasterio's transform_geom (fastest)
        return rasterio.warp.transform_geom(source_crs, target_crs, geometry)
    except Exception as e:
        logger.debug(f"Rasterio transform failed, trying OGR: {e}")
        
        # Fallback to OGR transformation if available
        if HAS_OGR:
            try:
                return _transform_geometry_ogr(geometry, source_crs, target_crs)
            except Exception as ogr_e:
                logger.warning(f"OGR transform also failed: {ogr_e}")
        
        logger.warning(f"Failed to transform geometry from {source_crs} to {target_crs}: {e}")
        return geometry


def _transform_geometry_ogr(geometry: Dict[str, Any], source_crs: str, target_crs: str) -> Dict[str, Any]:
    """Transform geometry using OGR."""
    # Create OGR geometry from GeoJSON
    geom = ogr.CreateGeometryFromJson(json.dumps(geometry))
    
    # Set up coordinate systems
    source_srs = osr.SpatialReference()
    target_srs = osr.SpatialReference()
    
    if source_crs.startswith('EPSG:'):
        source_srs.ImportFromEPSG(int(source_crs.split(':')[1]))
    else:
        source_srs.ImportFromWkt(source_crs)
        
    if target_crs.startswith('EPSG:'):
        target_srs.ImportFromEPSG(int(target_crs.split(':')[1]))
    else:
        target_srs.ImportFromWkt(target_crs)
    
    # Transform
    transform = osr.CoordinateTransformation(source_srs, target_srs)
    geom.Transform(transform)
    
    # Convert back to GeoJSON
    return json.loads(geom.ExportToJson())


def clip_array_with_cutline(data: np.ndarray, 
                           transform: rasterio.Affine,
                           geometries: List[Dict[str, Any]], 
                           crs: str = 'EPSG:3857',
                           nodata: float = np.nan) -> np.ndarray:
    """
    Clip a numpy array using cutline polygon geometries.
    
    If no geometries are provided (empty list), the entire array is masked as nodata.
    This handles the case where tiles are completely outside the cutline polygon.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array to clip
    transform : rasterio.Affine
        Affine transform of the data
    geometries : List[Dict[str, Any]]
        List of polygon geometries to use for clipping. Empty list = mask all as nodata.
    crs : str, optional
        Coordinate reference system of the data, defaults to 'EPSG:3857'
    nodata : float, optional
        Value to use for clipped areas, defaults to np.nan
        
    Returns
    -------
    np.ndarray
        Clipped data array with areas outside polygons set to nodata
    """
    # If no geometries provided, mask everything as nodata (tile outside cutline)
    if not geometries:
        return np.full_like(data, nodata)
        
    try:
        # Use geometry_mask for better performance with large geometries
        height, width = data.shape[-2:]
        
        # Create mask where True = inside polygons, False = outside
        mask = geometry_mask(
            geometries,
            out_shape=(height, width),
            transform=transform,
            invert=True  # Invert so True = inside polygons
        )
        
        # Apply mask to data
        result = data.copy()
        if result.ndim == 2:
            result[~mask] = nodata
        else:
            result[0][~mask] = nodata
            
        return result
                    
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


def clip_ogr_geometries_to_bounds(ogr_geometries: List['ogr.Geometry'], 
                                  bounds: Tuple[float, float, float, float]) -> List[Dict[str, Any]]:
    """
    Clip OGR geometries to the given bounds, returning GeoJSON dicts for rasterio.
    
    This avoids the inefficient JSON round-trip conversion by working with 
    OGR geometries directly until the final conversion.
    
    Parameters
    ----------
    ogr_geometries : List[ogr.Geometry]
        List of OGR geometry objects
    bounds : Tuple[float, float, float, float]
        Bounding box as (west, south, east, north)
        
    Returns
    -------
    List[Dict[str, Any]]
        List of clipped geometries as GeoJSON dicts for rasterio.mask
    """
    if not ogr_geometries:
        logger.debug("No OGR geometries provided to clip_ogr_geometries_to_bounds")
        return []
        
    if not HAS_OGR:
        logger.warning("OGR not available for geometry clipping")
        return []
        
    west, south, east, north = bounds
    clipped = []
    
    logger.debug(f"Clipping {len(ogr_geometries)} geometries to bounds: {bounds}")
    
    try:
        # Create bounding box geometry for clipping with proper coordinate system
        bbox_wkt = f"POLYGON(({west} {south}, {east} {south}, {east} {north}, {west} {north}, {west} {south}))"
        bbox_geom = ogr.CreateGeometryFromWkt(bbox_wkt)
        
        if not bbox_geom:
            logger.error(f"Failed to create bbox geometry from WKT: {bbox_wkt}")
            return []
            
        # Set the coordinate system to match the geometries (EPSG:3857)
        if HAS_OGR:
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(3857)  # Web Mercator
            bbox_geom.AssignSpatialReference(srs)
            logger.debug(f"Assigned EPSG:3857 to bbox geometry")
        
        for i, geom in enumerate(ogr_geometries):
            if not geom:
                logger.debug(f"Geometry {i} is None, skipping")
                continue
                
            if not geom.IsValid():
                logger.debug(f"Geometry {i} is invalid, skipping")
                continue
                
            # Debug coordinate systems
            geom_srs = geom.GetSpatialReference()
            bbox_srs = bbox_geom.GetSpatialReference()
            logger.debug(f"Geometry {i} SRS: {geom_srs.ExportToWkt()[:100] if geom_srs else 'None'}...")
            logger.debug(f"Bbox SRS: {bbox_srs.ExportToWkt()[:100] if bbox_srs else 'None'}...")
            
            # Get geometry bounds for comparison
            envelope = geom.GetEnvelope()  # Returns (minX, maxX, minY, maxY)
            geom_bounds = (envelope[0], envelope[2], envelope[1], envelope[3])  # (west, south, east, north)
            logger.info(f"BOUNDS COMPARISON:")
            logger.info(f"  Tile bounds:     {bounds}")
            logger.info(f"  Geometry bounds: {geom_bounds}")
            logger.info(f"  Tile bounds overlap geometry: west={bounds[0] < geom_bounds[2]}, south={bounds[1] < geom_bounds[3]}, east={bounds[2] > geom_bounds[0]}, north={bounds[3] > geom_bounds[1]}")
            
            # Check if bounding boxes overlap (basic test)
            bbox_overlap = (bounds[0] < geom_bounds[2] and bounds[2] > geom_bounds[0] and 
                          bounds[1] < geom_bounds[3] and bounds[3] > geom_bounds[1])
            logger.info(f"  Bounding box overlap: {bbox_overlap}")
                
            intersects = bbox_geom.Intersects(geom)
            logger.info(f"  OGR Intersects result: {intersects}")
            
            if bbox_overlap and not intersects:
                logger.warning(f"Bounding boxes overlap but OGR says no intersection - potential coordinate system issue!")
            
            if intersects:
                # Clip geometry to bounding box - this is the key optimization
                clipped_geom = geom.Intersection(bbox_geom)
                
                if clipped_geom and not clipped_geom.IsEmpty():
                    # Only convert to JSON at the very end
                    clipped_json_str = clipped_geom.ExportToJson()
                    if clipped_json_str:
                        clipped_dict = json.loads(clipped_json_str)
                        clipped.append(clipped_dict)
                        logger.debug(f"Successfully clipped geometry {i}")
                    else:
                        logger.debug(f"Failed to export clipped geometry {i} to JSON")
                else:
                    logger.debug(f"Clipped geometry {i} is empty or None")
                
    except Exception as e:
        logger.error(f"Failed to clip geometries to bounds: {e}. Using original geometries.")
        # Fallback: convert original geometries to dicts
        for i, geom in enumerate(ogr_geometries):
            try:
                if geom and geom.IsValid():
                    geom_json_str = geom.ExportToJson()
                    if geom_json_str:
                        geom_dict = json.loads(geom_json_str)
                        clipped.append(geom_dict)
                        logger.debug(f"Fallback: converted geometry {i} to dict")
            except Exception as fallback_e:
                logger.warning(f"Fallback failed for geometry {i}: {fallback_e}")
                continue
    
    logger.info(f"Clipped {len(ogr_geometries)} OGR geometries to {len(clipped)} parts within tile bounds")
    return clipped


def clip_geometries_to_bounds(geometries: List[Dict[str, Any]], 
                             bounds: Tuple[float, float, float, float]) -> List[Dict[str, Any]]:
    """
    Clip geometries to the given bounds, returning only the parts within the bounds.
    
    DEPRECATED: Use clip_ogr_geometries_to_bounds for better performance.
    """
    if not geometries or not HAS_OGR:
        return geometries
        
    west, south, east, north = bounds
    clipped = []
    
    try:
        # Create bounding box geometry for clipping
        bbox_wkt = f"POLYGON(({west} {south}, {east} {south}, {east} {north}, {west} {north}, {west} {south}))"
        bbox_geom = ogr.CreateGeometryFromWkt(bbox_wkt)
        
        for geom_dict in geometries:
            # Convert dict back to OGR geometry for spatial operations
            geom_json_str = json.dumps(geom_dict)
            geom = ogr.CreateGeometryFromJson(geom_json_str)
            
            if geom and bbox_geom.Intersects(geom):
                # Clip geometry to bounding box - this is the key optimization
                clipped_geom = geom.Intersection(bbox_geom)
                
                if clipped_geom and not clipped_geom.IsEmpty():
                    # Convert back to GeoJSON dict
                    clipped_json_str = clipped_geom.ExportToJson()
                    if clipped_json_str:
                        clipped_dict = json.loads(clipped_json_str)
                        clipped.append(clipped_dict)
                
    except Exception as e:
        logger.warning(f"Failed to clip geometries to bounds: {e}. Using original geometries.")
        return geometries
    
    logger.debug(f"Clipped {len(geometries)} geometries to {len(clipped)} parts within tile bounds")
    return clipped


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
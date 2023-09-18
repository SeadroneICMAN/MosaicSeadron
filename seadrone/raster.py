import rasterio
import cameratransform as ct
import numpy as np
import geopy
import imageio.v2 as imageio
import os
import glob

from pathlib import Path
from pandas import Series
from cameratransform import Camera
from pandas import DataFrame
from tqdm import tqdm
from typing import Tuple, List, Mapping, Any, Callable
from geopy.distance import geodesic as GD
from rasterio.transform import Affine
from rasterio.control import GroundControlPoint
from rasterio import DatasetReader
from fractions import Fraction
from numpy import ndarray, dtype
from geopy import Point

from seadrone.commons import search_raster_paths, get_first_file_extension
from seadrone.data_structures import Resampling

class GeorefenceUtils():
    """
        Class to solve Georefencing necessities
    """
    
    @staticmethod
    def get_transform(f : float, sensor_size : Tuple[float, float], image_size : Tuple[int, int], 
                      lat : float, lon : float, alt : float, yaw : float, 
                      pitch : float, roll : float) -> Affine:
        """
        Calculates a transformation matrix for a given capture in order to get every lat, lon for each pixel in the image.

        Args:
            f (float): focal_length
            sensor_size (Tuple[float, float]): correspondence pixel -> milimeter
            image_size (Tuple[int, int]): number of pixels for width and height
            lat (float): latitude of camera
            lon (float): longitude of camera
            alt (float): altitude of camera
            yaw (float): yaw of camera
            pitch (float): tilt of camera
            roll (float): roll of camera

        Returns:
            Affine: transformation matrix
        """

        cam : Camera = ct.Camera(ct.RectilinearProjection(focallength_mm = f, sensor = sensor_size, image = image_size),
                        ct.SpatialOrientation(elevation_m = alt, tilt_deg = pitch, roll_deg = roll, heading_deg = yaw, pos_x_m = 0, pos_y_m = 0))

        cam.setGPSpos(lat, lon, alt)

        coords : ndarray = np.array([cam.gpsFromImage([0, 0]), cam.gpsFromImage([image_size[0] - 1, 0]), cam.gpsFromImage([image_size[0] - 1, image_size[1] - 1]), cam.gpsFromImage([0, image_size[1] - 1])])
                
        gcp1 : GroundControlPoint = rasterio.control.GroundControlPoint(row = 0, col = 0, x = coords[0, 1], y = coords[0, 0], z = coords[0, 2])
        gcp2 : GroundControlPoint = rasterio.control.GroundControlPoint(row = image_size[0] - 1, col = 0, x = coords[1, 1], y = coords[1, 0], z = coords[1, 2])
        gcp3 : GroundControlPoint = rasterio.control.GroundControlPoint(row = image_size[0] - 1, col = image_size[1] - 1, x = coords[2, 1], y = coords[2, 0], z = coords[2, 2])
        gcp4 : GroundControlPoint = rasterio.control.GroundControlPoint(row = 0, col = image_size[1] - 1, x = coords[3, 1], y = coords[3, 0], z = coords[3, 2])

        return rasterio.transform.from_gcps([gcp1, gcp2, gcp3, gcp4])

    @staticmethod
    def get_georefence_by_uuid(metadata : DataFrame, 
                               lines : List[Mapping[str, int | float | None]] | None = None) -> Mapping[str, Affine]:
        """
        Given a DataFrame and a list of flight lines, calculate a dictionary with the transformation matrix for each capture

        Args:
            metadata (DataFrame): Pandas DataFrame that contains information like capture latitude, longitude, ...
            lines (List[Mapping[str, int | float | None]] | None, optional): List that indicates the flight lines. Defaults to [{'start': 0 , 'end': None , 'yaw': None, 'pitch': None, 'roll': None, 'alt' : None}].

        Returns:
            Mapping[str, Affine]: Dictionary that gathers captures IDs and transformation matrices
        """
        
        lines = lines if lines is not None else [{'start': 0 , 'end': None , 'yaw': None, 'pitch': None, 'roll': None, 'alt' : None}]

        georeference_by_uuid = {}

        for line in lines:
            captures : Series = metadata.iloc[line['start'] : line['end']]
            
            for _, capture in captures.iterrows():
                focal : float = capture['FocalLength']
                image_size : Tuple[int, int] = (capture['ImageWidth'], capture['ImageHeight'])[::-1]
                sensor_size : Tuple[float, float] = (capture['SensorX'], capture['SensorY'])[::-1]

                lat : float = float(capture['GPSLatitude'])
                lon : float = float(capture['GPSLongitude'])
                alt : float = float(capture['GPSAltitude']) if line['alt'] is None else line['alt']
                pitch : float = float(capture['Pitch']) if line['pitch'] is None else line['pitch']
                roll : float = float(capture['Roll']) if line['roll'] is None else line['roll']
                yaw : float = float(capture['Yaw']) if line['yaw'] is None else line['yaw']

                georeference_by_uuid[os.path.basename(capture['ID'])] = GeorefenceUtils.get_transform(f = focal, sensor_size = sensor_size, image_size = image_size,
                                                        lat = lat, lon = lon, alt = alt, yaw = yaw, pitch = pitch, roll = roll)

        return georeference_by_uuid

    @staticmethod
    def georeference_bands(georefence_by_uuid : Mapping[str, Affine], in_folder : str, out_folder : str, 
                           profile : Mapping[str, Any], flip_axis : List[int] = None, 
                           overwrite : bool = False) -> None:
        """
        Given the correspondence Capture ID -> Transformation Matrix, georefence every capture in the in_folder based on the georefence_by_uuid parameter.

        Args:
            georefence_by_uuid (Mapping[str, Affine]): Dictionary with IDs -> transformation matrix for each capture
            in_folder (str): folder path of the captures to georefence
            out_folder (str): folder path where to save the georefenced captures
            profile (Mapping[str, Any]): Metadata of the georefenced captures
            flip_axis (List[int]): Which axis will be flipped
            overwrite (bool, optional): Overwrite or not. Defaults to False.
        """

        os.makedirs(out_folder, exist_ok = True)
        
        for uuid, transform in tqdm(georefence_by_uuid.items(), total = len(georefence_by_uuid.items())):
            profile['transform'] = transform
            source = uuid.split('.')
            source[-1] = get_first_file_extension(in_folder)
            source = ''.join(source)

            if not os.path.exists(os.path.join(out_folder, uuid)) or overwrite:
                with rasterio.open(os.path.join(in_folder, source), 'r') as src:       
                    with rasterio.open(os.path.join(out_folder, uuid), 'w', **profile) as dst:
                        data : ndarray = src.read().astype(profile['dtype'])
                        dst.write( data if flip_axis is None else np.flip(data, axis = flip_axis) )
        
    @staticmethod
    def georeference_bands_with_metadata(metadata : DataFrame, georefence_by_uuid : Mapping[str, Affine], 
                                         in_folder : str, out_folder : str, profile : Mapping[str, Any], 
                                         flip_axis : List[int] = None, overwrite : str = False) -> None:
        """
        Given the correspondence Capture ID -> Transformation Matrix, georefence every capture in the in_folder based on the georefence_by_uuid parameter.
        The additional metadata parameter is used to obtain the names of the georefenced captures.

        Args:
            metadata (DataFrame): Pandas DataFrame that contains information like capture latitude, longitude, ...
            georefence_by_uuid (Mapping[str, Affine]): Dictionary with IDs -> transformation matrix for each capture
            in_folder (str): folder path of the captures to georefence
            out_folder (str): folder path where to save the georefenced captures
            profile (Mapping[str, Any]): Metadata of the georefenced captures
            flip_axis (List[int]): Which axis will be flipped
            overwrite (bool, optional): Overwrite or not. Defaults to False.
        """

        os.makedirs(out_folder, exist_ok = True)
        metadata = metadata.set_index(metadata['ID'])
        for uuid, transform in tqdm(georefence_by_uuid.items(), total = len(georefence_by_uuid.items())):
            source : str = os.path.basename(metadata['Source'][uuid]).split('.')
            source[-1] = get_first_file_extension(in_folder)
            source = ''.join(source)
            profile['transform'] = transform

            if not os.path.exists(os.path.join(out_folder, uuid)) or overwrite:
                with rasterio.open(os.path.join(in_folder, source), 'r') as src:       
                    with rasterio.open(os.path.join(out_folder, uuid), 'w', **profile) as dst:
                        data : ndarray = src.read().astype(profile['dtype'])
                        dst.write( data if flip_axis is None else np.flip(data, axis = flip_axis) )

    @staticmethod
    def georeference_images(georefence_by_uuid : Mapping[str, Affine], in_folder : str, out_folder : str, 
                            profile : Mapping[str, Any], flip_axis : List[int] = None, overwrite : bool = False) -> None:
        """
        Given the correspondence Capture ID -> Transformation Matrix, georefence every capture in the in_folder based on the georefence_by_uuid parameter.
        The difference between 'georefence_bands' and this method is that this method reads an image and georefence that image.
        In the method 'georefence_bands', we spect to read a raster file and georefence all its bands in the same file.

        Args:
            georefence_by_uuid (Mapping[str, Affine]): Dictionary with IDs -> transformation matrix for each capture
            in_folder (str): folder path of the captures to georefence
            out_folder (str): folder path where to save the georefenced captures
            profile (Mapping[str, Any]): Metadata of the georefenced captures
            flip_axis (List[int]): Which axis will be flipped
            overwrite (bool, optional): Overwrite or not. Defaults to False.
        """

        os.makedirs(out_folder, exist_ok = True)
        
        for uuid, transform in tqdm(georefence_by_uuid.items(), total = len(georefence_by_uuid.items())):
            profile['transform'] = transform
            source = uuid.split('.')
            source[-1] = get_first_file_extension(in_folder)
            source = ''.join(source)

            if not os.path.exists(os.path.join(out_folder, uuid)) or overwrite:
                with rasterio.open(os.path.join(out_folder, uuid), 'w', **profile) as dst:
                    img : ndarray = imageio.imread(os.path.join(in_folder, source))
                    img = np.array([ img[:, :, i] for i in range(profile['count'])]).astype(profile['dtype'])
                    dst.write( img if flip_axis is None else np.flip(img, axis = flip_axis) )

    @staticmethod
    def georeference_images_with_metadata(metadata : DataFrame, georefence_by_uuid : Mapping[str, Affine], 
                                          in_folder : str, out_folder : str, profile : Mapping[str, Any], 
                                          flip_axis : List[int] = None, overwrite : bool = False) -> None:
        """
        Given the correspondence Capture ID -> Transformation Matrix, georefence every capture in the in_folder based on the georefence_by_uuid parameter.
        The additional metadata parameter is used to obtain the names of the georefenced captures.
        The difference between 'georeference_images_with_metadata' and this method is that this method reads an image and georefence that image.
        In the method 'georeference_images_with_metadata', we spect to read a raster file and georefence all its bands in the same file.

        Args:
            metadata (DataFrame): Pandas DataFrame that contains information like capture latitude, longitude, ...
            georefence_by_uuid (Mapping[str, Affine]): Dictionary with IDs -> transformation matrix for each capture
            in_folder (str): folder path of the captures to georefence
            out_folder (str): folder path where to save the georefenced captures
            profile (Mapping[str, Any]): Metadata of the georefenced captures
            flip_axis (List[int]): Which axis will be flipped
            overwrite (bool, optional): Overwrite or not. Defaults to False.
        """

        os.makedirs(out_folder, exist_ok = True)
        metadata = metadata.set_index(metadata['ID'])

        for uuid, transform in tqdm(georefence_by_uuid.items(), total = len(georefence_by_uuid.items())):
            source : str = os.path.basename(metadata['Source'][uuid]).split('.')
            source[-1] = get_first_file_extension(in_folder)
            source = ''.join(source)
            profile['transform'] = transform

            if not os.path.exists(os.path.join(out_folder, uuid)) or overwrite:
                with rasterio.open(os.path.join(out_folder, uuid), 'w', **profile) as dst:
                    img : ndarray = imageio.imread(os.path.join(in_folder, source))
                    img = np.array([ img[:, :, i] for i in range(profile['count'])]).astype(profile['dtype'])
                    dst.write( img if flip_axis is None else np.flip(img, axis = flip_axis) )
        

class RasterUtils():
    """A class that gathers some useful method such as split the bands of a raster into multiple file, ...
    """

    @staticmethod
    def open_raster(raster_path : str) -> np.array:
        """This function receives a raster path and returns its data using rasterio

        Args:
            raster_path (str): raster path

        Returns:
            np.array: data
        """

        data : np.array = None

        with rasterio.open(raster_path, 'r') as src:
            data = src.read()

        return data
    
    @staticmethod
    def load_images_from_list(img_list : List[str]) -> List[np.array]:
        """This function receives a list of images and returns its data using rasterio

        Args:
            img_list (List[str]): a list of images

        Returns:
            List[np.array]: a list of data images
        """
        return [RasterUtils.open_raster(raster_path) for raster_path in img_list]
    
    @staticmethod
    def load_images(img_folder : str) -> List[np.array]:
        """This functions receives a folder and return a list with all the data of all the images contained using rasterio

        Args:
            img_folder (str): image folder

        Returns:
            List[np.array]: a list data images
        """
        return [RasterUtils.open_raster(raster_path) for raster_path in glob.glob(os.path.join(img_folder, '*'))]

    @staticmethod
    def resample(in_folder : str, out_folder : str, resamplings : List[Resampling]):
        """
        This function receives an in_folder for reading, an out_folder for writing and 
        a resamplings list to apply different resampling settings

        Args:
            in_folder (str): where the files to resample are
            in_folder (str): where to save the resampled files
            resamplings (List[Resampling]): a list of Resampling settings
        """
        
        raster_paths = search_raster_paths(in_folder)
        
        for raster_path in raster_paths:
            diff = os.path.relpath(raster_path.replace('.tif', '{}.tif'), out_folder)
            diff = str(Path(*[part for part in Path(diff).parts if part != '..'][1:]))
            out_path = os.path.join(out_folder, diff)

            os.makedirs(os.path.dirname(out_path), exist_ok = True)

            with rasterio.open(raster_path, 'r') as dataset:
                for resampling in resamplings:
                    
                    data : ndarray = dataset.read(
                        out_shape = (
                            dataset.count,
                            int(dataset.height * resampling.scale_x),
                            int(dataset.width * resampling.scale_y)
                        ),
                        resampling = resampling.method
                    )
                    
                    dst_transform : Affine = dataset.transform * dataset.transform.scale(
                        (dataset.width / data.shape[-1]),
                        (dataset.height / data.shape[-2])
                    )
                    
                    dst_kwargs : Mapping[str, Any] = dataset.meta.copy()
                    dst_kwargs.update(
                        {
                            "crs": dataset.crs,
                            "transform": dst_transform,
                            "width": data.shape[-1],
                            "height": data.shape[-2],
                        }
                    )
                    
                    with rasterio.open(out_path.format(f'_{resampling.id}'), "w", **dst_kwargs) as dst:
                        dst.write(data)

    @staticmethod
    def get_raster_corners(raster_path : str) -> List[Tuple[float, float]]:
        """
        Given a raster path, return a list of its corners based on its transformation matrix.

        Args:
            raster_path (str): path of the raster to be processed

        Returns:
            List[Tuple[float, float]]: List with the 4 corners of the raster
        """

        raster : DatasetReader = rasterio.open(raster_path)
        w, h = raster.width, raster.height

        return [raster.transform * p for p in [(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)]]
    
    @staticmethod
    def get_raster_corners_by_params(transform : Affine, width : int, 
                                     height : int) -> List[Tuple[float, float]]:
        """
        Given a transformation matrix, a width and a height, return a list of corners based on the given transformation matrix.

        Args:
            transform (Affine): transformation matrix
            width (int): transformation width
            height (int): transformation height

        Returns:
            List[Tuple[float, float]]: List with the 4 corners of the raster
        """

        return [transform * p for p in [(0, 0), (0, height - 1), (width - 1, height - 1), (width - 1, 0)]]

    @staticmethod
    def split_bands(raster_path : str, 
                    band_names : List[str] | None = None):
        """A function that receives a raster path, a list of bands names and splits its bands into one tif file for each band

        Args:
            raster_path (str): raster path
            band_names (List[str] | None, optional): a list of band names. Defaults to None.
        """

        with rasterio.open(raster_path) as src:
            band_names = band_names if band_names is not None else [f'{i}' for i in range(src.profile['count'])]
            data = src.read()
            profile = src.profile
            profile['count'] = 1
            for band_idx in range(src.profile['count']):
                out_folder = os.path.join(os.path.dirname(raster_path), band_names[band_idx])
                os.makedirs(out_folder, exist_ok = True)

                out_name = os.path.basename(raster_path)
                
                with rasterio.open(os.path.join(out_folder, out_name), 'w', **profile) as dst:
                    dst.write(data[band_idx][np.newaxis, :])

    @staticmethod
    def split_bands_batch(raster_paths : List[str], 
                          band_names : List[str] | None = None):
        """This function receives a list of raster_paths and a list of band_names. It splits each band of each raster into
        one tif tile for each band using the specified band_names

        Args:
            raster_paths (List[str]): _description_
            band_names (List[str] | None, optional): _description_. Defaults to None.
        """

        for raster in raster_paths:
           RasterUtils.split_bands(raster, band_names)

    @staticmethod
    def join_bands(band_paths : List[str], out_folder : str):
        """This function receives a list of paths representing a band and a folder to save the stacked version

        Args:
            band_paths (List[str]): a list of paths representing a band
            out_folder (str): where to save the stacked version
        """

        out_name = os.path.basename(band_paths[0])

        data = []
        for band in band_paths:
            with rasterio.open(band) as src:
                profile = src.profile
                data.append(src.read(1))
        else:
            data = np.array(data)
            profile['count'] = len(band_paths)

        with rasterio.open(os.path.join(out_folder, out_name), 'w', **profile) as dst:
            dst.write(data)

    @staticmethod
    def join_bands_batch(raster_paths : List[List[str]], out_folder : str) -> None:
        """This function receives a list of lists of paths representing a band and a folder to save the stacked version

        Args:
            raster_paths (List[List[str]]): a list of lists of paths representing a band
            out_folder (str): where to save the stacked version
        """

        for band_paths in raster_paths:
            RasterUtils.join_bands(band_paths, out_folder)


class GeometryUtils():
    """This class gathers geometry methos suchs get the center of a list of points or if a point is within a polygon, ...
    """

    @staticmethod
    def is_on_right_side(x : float, y : float, xy0 : Tuple[float, float], xy1 : Tuple[float, float]) -> bool:
        """
        Given a point and 2 points defining a rect, check if the point is on the right side or not.        

        Args:
            x (float): value in the x-axis of the point
            y (float): value in the y-axis of the point
            xy0 (Tuple[float, float]): point 0 of the rect
            xy1 (Tuple[float, float]): point 1 of the rect

        Returns:
            bool: is on right side or not
        """

        x0, y0 = xy0
        x1, y1 = xy1
        a : float = float(y1 - y0)
        b : float = float(x0 - x1)
        c : float = - a * x0 - b * y0
        return a * x + b * y + c > 0

    @staticmethod
    def is_point_within_vertices(x : float, y : float, vertices : List[Tuple[float, float]]) -> bool:
        """This fuction checks if a point is within the given vertices

        Args:
            x (float): value in the width axis for the point
            y (float): value in the height axis for the point
            vertices (List[Tuple[float, float]]): bounding vertices

        Returns:
            bool: whether the point is within the vertices or not
        """

        num_vert : int = len(vertices)
        is_right : bool = [GeometryUtils.is_on_right_side(x, y, vertices[i], vertices[(i + 1) % num_vert]) for i in range(num_vert)]
        all_left : bool = not any(is_right)
        all_right : bool = all(is_right)
        return all_left or all_right

    @staticmethod
    def are_points_within_vertices(vertices : List[Tuple[float, float]], points : List[Tuple[float, float]]) -> bool:
        """
        Given a list of vertices and a list of points, generate every rect determined by the vertices and check if the points are within the polygon or not.

        Args:
            vertices (List[Tuple[float, float]]): List of vertices defining a polygon
            points (List[Tuple[float, float]]): List of points to study is they are within the polygon or not

        Returns:
            bool: the given points are within the given vertices or not
        """

        all_points_in_merge : bool = True

        for point in points:
            all_points_in_merge &= GeometryUtils.is_point_within_vertices(x = point[0], y = point[1], vertices = vertices)
        
        return all_points_in_merge

    @staticmethod
    def euclidean_distance(p1 : Tuple[float, float], p2 : Tuple[float, float]) -> float:
        """
        euclidean distance between two points

        Args:
            p1 (Tuple[float, float]): 2D point 1
            p2 (Tuple[float, float]): 2D point 2

        Returns:
            float: euclidean distance between two points
        """

        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    @staticmethod
    def get_center(points : np.ndarray) -> np.ndarray:
        """This function receives a list of points and returns the point at the center of all points

        Args:
            points (np.ndarray): a list of points

        Returns:
            np.ndarray: center of all points
        """

        x = points[:, 0]
        y = points[:, 1]

        m_x = sum(x) / points.shape[0]
        m_y = sum(y) / points.shape[0]

        return np.array([m_x, m_y])


class Paralelogram2D:
    """This function represents a paralelogram
    """

    def __init__(self, points : List[Tuple[float, float]]):
        """This constructor receives a list of points and sets the pairs of lines

        Args:
            points (List[Tuple[float, float]]): _description_
        """

        self.points = points
        self.lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
        self.pairs = [[0, 2], [1, 3]]

    def get_line_center(self, index : int) -> np.ndarray:
        """This functions returns the center of a specific line of the paralelogram

        Args:
            index (int): line index

        Returns:
            np.ndarray: center
        """

        return sum(self.points[self.lines[index]]) / 2
    
    def get_offset_to_lines(self, index : int, point : np.ndarray) -> np.ndarray:
        """This functions returns a Vector that represents what should be direction of point for being in the specified line

        Args:
            index (int): line index
            point (np.ndarray): point

        Returns:
            np.ndarray: direction vector
        """

        return self.get_line_center(index) - point
    
    def get_center(self) -> np.ndarray:
        """This function returns the center of the paralelogram

        Returns:
            np.ndarray: center
        """

        return GeometryUtils.get_center(self.points)
    
    def move_line_from_offset(self, index : int, offset : np.ndarray):
        """This function moves a specific line given an offset vector

        Args:
            index (int): line index
            offset (np.ndarray): offset vector
        """

        self.points[self.lines[index]] += offset
    
    def are_on_right_side_of_line(self, index : int, points : np.ndarray) -> bool:
        """This function checks if a list of points is on the right side of a specific line

        Args:
            index (int): line index
            points (np.ndarray): a list of points

        Returns:
            bool: whether the list is on the right side or not
        """

        return all([GeometryUtils.is_on_right_side(*point, *self.points[self.lines[index]]) for point in points])


class MergeUtils():

    @staticmethod    
    def _latlon_to_index(dst , src : DatasetReader) -> ndarray:
        """
        Given a source dataset and a destination dataset. Get the latitudes and longitudes that correspond to move the source data to the destination data.

        Args:
            dst (_type_): Destination dataset
            src (DatasetReader): Source dataset

        Returns:
            ndarray: List of latitudes and longitudes
        """

        cols, rows = np.meshgrid(np.arange(src.width), np.arange(src.height))
        
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        lons, lats = np.array(xs), np.array(ys)

        coords_to_index = np.array([ dst.index(lons[i], lats[i]) for i in np.arange(src.height)])
        lons, lats = coords_to_index[:, 0, :], coords_to_index[:, 1, :]
        
        return lons, lats

    @staticmethod
    def _is_merge_valid(corners : List[Tuple[float, float]], points : List[Tuple[float, float]]) -> bool:
        """
        Given a list of points and a list of corners, return if all the points are within the corners.

        Args:
            corners (List[Tuple[float, float]]): corners of a transformation matrix
            points (List[Tuple[float, float]]): points to check if they are within the corners

        Returns:
            bool: return all the points are within the corners
        """

        all_points_in_merge : bool = True

        for point in points:
            all_points_in_merge &= GeometryUtils.is_within_vertices(x = point[0], y = point[1], vertices = corners)
        
        return all_points_in_merge

    @staticmethod
    def _get_merge_transform(raster_paths : set, max_iterations : int = 2000) -> Tuple[int, int, Affine]:
        """This function returns a transform matrix that contains of the specified rasters

        Args:
            raster_paths (set): raster paths to merge
            max_iterations (int, optional): additional merge parameters. Default is 2000

        Returns:
            Tuple[int, int, Affine]: width, height and transformation matrix of the merge
        """
        

        with rasterio.open(raster_paths[0]) as src:
            original_transform : Affine = src.transform
            transform : Affine = src.transform
            width : int = src.width
            height : int = src.height
            res : tuple = src.res
        
        for raster_path in raster_paths:
            with rasterio.open(raster_path) as src:
                if res[0] < src.res[0]:
                    original_transform : Affine = src.transform
                    transform : Affine = src.transform
                    res : tuple = src.res

        
        raster_corners : ndarray = np.array([RasterUtils.get_raster_corners(raster_path = raster_path) for raster_path in raster_paths]).reshape(-1, 2)

        mid_point = GeometryUtils.get_center(raster_corners)
        mid_point_first_capture = GeometryUtils.get_center(raster_corners[0 : 4])
        c, f = mid_point[0] + (raster_corners[0][0] - mid_point_first_capture[0]), mid_point[1] + (raster_corners[0][1] - mid_point_first_capture[1])

        transform = Affine(a = original_transform.a,
                        b = original_transform.b,
                        c = c,
                        d = original_transform.d,
                        e = original_transform.e,
                        f = f,
                        )
        
        paralelo = Paralelogram2D( np.array( RasterUtils.get_raster_corners_by_params(transform, width, height) ))
    
        for line_index in range(len(paralelo.lines)):
            offset = paralelo.get_offset_to_lines(line_index, paralelo.get_center())

            iteration = 0
            while not paralelo.are_on_right_side_of_line(line_index, raster_corners) and iteration < max_iterations:
                paralelo.move_line_from_offset(line_index, offset)
                iteration += 1

        width : int = int(round(GeometryUtils.euclidean_distance(paralelo.points[0], paralelo.points[-1]) / res[0]))
        height : int = int(round(GeometryUtils.euclidean_distance(paralelo.points[0], paralelo.points[1]) / res[1]))
        
        transform = Affine(a = original_transform.a,
                        b = original_transform.b,
                        c = paralelo.points[0][0],
                        d = original_transform.d,
                        e = original_transform.e,
                        f = paralelo.points[0][1])
        
        return width, height, transform

    @staticmethod
    def _mean(dst , raster_paths : List[str], n_bands : int, width : int, height : int, dtype : dtype = np.float32, band_index : int | None = None) -> ndarray:
        """
        Merge method that calculates the mean value in those positions where more than one raster write its values.

        Args:
            dst (_type_): destination raster
            raster_paths (List[str]): raster paths to merge
            n_bands (int): bands of each raster
            width (int): width of the merge raster
            height (int): height of the merge raster
            dtype (dtype, optional): dtype of the merge raster. Defaults to np.float32.
            band_index (int | None, optional): if not None we only merge the specified band. Defaults to None.

        Returns:
            ndarray: resulting merge
        """

        final_data : ndarray = np.zeros(shape = (n_bands, height, width), dtype = dtype)
        count : ndarray = np.zeros(shape = (n_bands, height, width), dtype = np.uint8)

        for raster_path in tqdm(raster_paths):
            with rasterio.open(raster_path, 'r') as src:
                data : ndarray = src.read() if band_index is None else np.array([src.read(band_index)])
                
                lons, lats = MergeUtils._latlon_to_index(dst, src)

                final_data[:, lons, lats] = np.nansum([data, final_data[:, lons, lats]], axis = 0)
                count[:, lons, lats] = np.nansum([~np.isnan(data), count[:, lons, lats]], axis = 0)
                
        return np.divide(final_data, count)
    
    @staticmethod
    def _first(dst , raster_paths : List[str], n_bands : int, width : int, height : int, dtype : dtype = np.float32, band_index : int | None = None) -> ndarray:
        """
        Merge method that keeps the first value in write those positions where more than one raster write its values.

        Args:
            dst (_type_): destination raster
            raster_paths (List[str]): raster paths to merge
            n_bands (int): bands of each raster
            width (int): width of the merge raster
            height (int): height of the merge raster
            dtype (dtype, optional): dtype of the merge raster. Defaults to np.float32.
            band_index (int | None, optional): if not None we only merge the specified band. Defaults to None.

        Returns:
            ndarray: resulting merge
        """

        final_data : ndarray = np.empty(shape = (n_bands, height, width), dtype = dtype)
        final_data[:] = np.NaN

        for raster_path in tqdm(raster_paths):
            with rasterio.open(raster_path, 'r') as src:
                data : ndarray = src.read() if band_index is None else np.array([src.read(band_index)])

                lons, lats = MergeUtils._latlon_to_index(dst, src)
                
                dst_arr = final_data[:, lons, lats]
                np.copyto(dst_arr, data, where = np.isnan(dst_arr) * ~np.isnan(data))
                final_data[:, lons, lats] = dst_arr
                
        return final_data

    @staticmethod
    def _max(dst , raster_paths : List[str], n_bands : int, width : int, height : int, dtype : dtype = np.float32, band_index : int | None = None) -> ndarray:
        """
        Merge method that calculates the max value in those positions where more than one raster write its values.

        Args:
            dst (_type_): destination raster
            raster_paths (List[str]): raster paths to merge
            n_bands (int): bands of each raster
            width (int): width of the merge raster
            height (int): height of the merge raster
            dtype (dtype, optional): dtype of the merge raster. Defaults to np.float32.
            band_index (int | None, optional): if not None we only merge the specified band. Defaults to None.

        Returns:
            ndarray: resulting merge
        """

        final_data : ndarray = np.empty(shape = (n_bands, height, width), dtype = dtype)
        final_data[:] = np.NaN

        for raster_path in tqdm(raster_paths):
            with rasterio.open(raster_path, 'r') as src:
                data : ndarray = src.read() if band_index is None else np.array([src.read(band_index)])

                lons, lats = MergeUtils._latlon_to_index(dst, src)
                
                final_data[:, lons, lats] = np.nanmax([data, final_data[:, lons, lats]], axis = 0)
                
        return final_data

    @staticmethod
    def _min(dst , raster_paths : List[str], n_bands : int, width : int, height : int, dtype : dtype = np.float32, band_index : int | None = None) -> ndarray:
        """
        Merge method that calculates the min value in those positions where more than one raster write its values.

        Args:
            dst (_type_): destination raster
            raster_paths (List[str]): raster paths to merge
            n_bands (int): bands of each raster
            width (int): width of the merge raster
            height (int): height of the merge raster
            dtype (dtype, optional): dtype of the merge raster. Defaults to np.float32.
            band_index (int | None, optional): if not None we only merge the specified band. Defaults to None.

        Returns:
            ndarray: resulting merge
        """

        final_data : ndarray = np.empty(shape = (n_bands, height, width), dtype = dtype)
        final_data[:] = np.NaN

        for raster_path in tqdm(raster_paths):
            with rasterio.open(raster_path, 'r') as src:
                data : ndarray = src.read() if band_index is None else np.array([src.read(band_index)])

                lons, lats = MergeUtils._latlon_to_index(dst, src)
                
                final_data[:, lons, lats] = np.nanmin([data, final_data[:, lons, lats]], axis = 0)
                
        return final_data

    @staticmethod
    def merge(raster_paths : List[str], out_name : str, method : str = 'mean', dtype : dtype = np.float32, band_names : List[str] | None = None, **kwargs) -> str:
        """This function merges all the given rasters into a single raster file 

        Args:
            raster_paths (List[str]): raster paths to merge
            out_name (str): name of the merge raster
            method (str, optional): merge method [mean, first, min, max]. Defaults to 'mean'.
            dtype (dtype, optional): dtype of the merge raster. Defaults to np.float32.
            band_names (List[str] | None, optional): List of band names. If it is not None, write one file for each band instead of one file with all the bands. Defaults to None.

        Returns:
            str: name of the merge raster
        """
        
        methods : Mapping[str, Callable] = {
            'mean' : MergeUtils._mean,
            'first' : MergeUtils._first,
            'max' : MergeUtils._max,
            'min' : MergeUtils._min,
        }

        method : Callable = methods.get(method, method)

        with rasterio.open(raster_paths[0], 'r') as raster:
            n_bands : int = raster.count
            profile : Mapping[str, Any] = raster.profile
            if len(raster_paths) > 1:
                width, height, transform = MergeUtils._get_merge_transform(raster_paths = raster_paths, **kwargs)
                profile['width'] = width
                profile['height'] = height
                profile['transform'] = transform
            else:
                width, height = raster.width, raster.height

        if band_names is not None and n_bands == len(band_names):
            profile['count'] = 1
            
            with rasterio.open(out_name.replace('.', f'_band_{band_names[0]}.'), 'w', **profile) as dst:
                data : ndarray = method(dst, raster_paths, n_bands, width, height, dtype)

            for band_index in range(n_bands):
                with rasterio.open(out_name.replace('.', f'_band_{band_names[band_index]}.'), 'w', **profile) as dst:
                    dst.write( np.array([data[band_index]]) )
        else:
            with rasterio.open(out_name, 'w', **profile) as dst:
                dst.write( method(dst, raster_paths, n_bands, width, height, dtype) )
            
        return out_name
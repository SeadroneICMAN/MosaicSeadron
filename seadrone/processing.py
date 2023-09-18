import pandas as pd
import numpy as np
import yaml
import os
import rasterio
import spectral
import math
import cv2

from functools import cache
from tqdm import tqdm
from typing import Tuple, List, Mapping, Any
from dataclasses import asdict
from pyexiv2 import Image

from micasense import imageutils, imageset, capture
from micasense.image import Image as Image_micasense
from seadrone.raster import GeorefenceUtils, MergeUtils, RasterUtils
from seadrone.enums import FlightMode, SensorType
from seadrone.data_structures import GeoreferencePartition, MergePartition, Sensor, Resampling


class FlightProcessor:
    """This class gathers all the required methods to process a flight
    """

    @staticmethod
    def get_captures_metadata(in_folder : str) -> pd.DataFrame:
        """
        Read the images from the in_folder and pd.DataFrame with the metadata of each image

        Args:
            in_folder (str): folder where to read the metadata file

        Returns:
            pd.DataFrame: metadata for each image
        """

        lines : List[str] = []
        image_paths : List[str] = [image_path for image_path in os.listdir(in_folder)]

        for uuid, image_path in tqdm(enumerate(image_paths), total = len(image_paths)):
            image_path : str = os.path.join(in_folder, image_path)
            metadata = FlightProcessor.get_capture_metadata(uuid, image_path)
            lines.append(list(metadata.values()))
        else:
            header = list(metadata.keys())

        return pd.DataFrame(columns = header, data = lines)
    
    @staticmethod
    def get_capture_metadata(uuid : str, image_path : str) -> Mapping[str, Any]:
        """Function that read a file and extract its metadata values in a dict

        Args:
            uuid (str): uuid
            image_path (str): full path to the image

        Returns:
            Mapping[str, Any]: metadata values
        """

        def reduce_values(values : List[str]) -> List[float]:
            return [ float(i.split('/')[0]) / float(i.split('/')[1]) for i in values]

        def decimal_coords(coords : str, ref : str) -> float:
            coords : List[float] = reduce_values(coords.split(' '))

            decimal_degrees : float = coords[0] + coords[1] / 60 + coords[2] / 3600

            if ref == "S" or ref =='W' :
                decimal_degrees = -decimal_degrees
            
            return decimal_degrees
        
        FOCAL_LENGTH : str = 'Exif.Photo.FocalLength'
        GPS_DATETIME : str = 'Exif.Photo.DateTimeOriginal'
        GPS_ALTITUDE : str = 'Exif.GPSInfo.GPSAltitude'
        GPS_LONGITUDE : str = 'Exif.GPSInfo.GPSLongitude'
        GPS_LONGITUDE_REF : str = 'Exif.GPSInfo.GPSLongitudeRef'
        GPS_LATITUDE : str = 'Exif.GPSInfo.GPSLatitude'
        GPS_LATITUDE_REF : str = 'Exif.GPSInfo.GPSLatitudeRef'
        YAW : str = 'Xmp.drone-dji.FlightYawDegree'
        PITCH : str = 'Xmp.drone-dji.FlightPitchDegree'
        ROLL : str = 'Xmp.drone-dji.FlightRollDegree'
        IMAGE_WIDTH : str = 'Exif.Photo.PixelXDimension'
        IMAGE_HEIGHT : str = 'Exif.Photo.PixelYDimension'
        X_RESOLUTION : str = 'Exif.Thumbnail.XResolution'
        Y_RESOLUTION : str = 'Exif.Thumbnail.YResolution'
        RESOLUTION_UNITS : str = 'Exif.Thumbnail.ResolutionUnit'

        image : Image = Image(image_path)

        exif_metadata : Mapping[str, Any] = image.read_exif()
        xmp_metadata : Mapping[str, Any]= image.read_xmp()

        focal : float = reduce_values([exif_metadata.get(FOCAL_LENGTH, '13500/1000')])[0]
        lon_ref, lat_ref = exif_metadata.get(GPS_LONGITUDE_REF), exif_metadata.get(GPS_LATITUDE_REF)

        datestamp, timestamp = exif_metadata.get(GPS_DATETIME).split(' ')

        lon, lat, alt = decimal_coords(exif_metadata.get(GPS_LONGITUDE), lon_ref), decimal_coords(exif_metadata.get(GPS_LATITUDE), lat_ref), reduce_values([exif_metadata.get(GPS_ALTITUDE)])[0]
            
        width, height = int(exif_metadata.get(IMAGE_WIDTH)), int(exif_metadata.get(IMAGE_HEIGHT))
            
        x_res : float = reduce_values([exif_metadata.get(X_RESOLUTION)])[0]
        y_res : float = reduce_values([exif_metadata.get(Y_RESOLUTION)])[0]
            
        if int(exif_metadata.get(RESOLUTION_UNITS)) == 2:
            x_res = width * (25.4 / x_res)
            y_res = height * (25.4 / y_res)

        sensor_size : Tuple[float, float] = width / x_res, height / y_res

        yaw, pitch, roll = angles = float(xmp_metadata.get(YAW)) % 360, float(xmp_metadata.get(PITCH)) % 360, float(xmp_metadata.get(ROLL)) % 360, 

        data = {
                'Source' : image_path,
                'ID' : str(uuid) + '.tif',
                'GPSDateStamp' : datestamp.replace(':', '-'),
                'GPSTimeStamp' : timestamp,
                'GPSLatitude' : lat,
                'GPSLatitudeRef' : lat_ref,
                'GPSLongitude' : lon,
                'GPSLongitudeRef' : lon_ref,
                'GPSAltitude' : alt,
                'SensorX' : sensor_size[0],
                'SensorY' : sensor_size[1],
                'FocalLength' : focal,
                'Yaw' : yaw,
                'Pitch' : pitch,
                'Roll' : roll,
                'SolarElevation' : 0,
                'ImageWidth' : width,
                'ImageHeight' : height,
                'XResolution' : x_res,
                'YResolution' : y_res,
                'ResolutionUnits' : 'mm',
            }
        
        return data

    @staticmethod
    def compute_lines(lines : List[Tuple[int, int]], indexes : List[int], 
                      start : int = 0, end : int = 0) -> List[int]:
        """A function that given a list of indexes where there are gaps, 
        returns a list of pairs(start, end) for each interval

        Args:
            lines (List[Tuple[int, int]]): list where to write the result
            indexes (List[int]): list of indexes
            start (int, optional): first index. Defaults to 0.
            end (int, optional): last index. Defaults to 0.

        Returns:
            List[int]: list of pairs(start, end) for each interval
        """

        for index in indexes:
            if abs(end - index) > 1:
                if start != end:
                    lines.append((int(start), int(end)))
                start = index
            end = index
        else:
            if start != end:
                lines.append((int(start), int(end)))
        
        return list(set(lines))
    
    @staticmethod
    def compute_flight_lines(captures_yaw : pd.Series, altitude : float, yaw : float, 
                             pitch : float, roll : float, flight_mode : FlightMode, 
                             error : float = 10) -> List[Mapping[str, int | float | None]]:
        """A function that receives a list a yaws and flight values such as altitude.
        With this, it extracts those captures that are not or could not be turns because its yaw 
        is similar to previous captures

        Args:
            captures_yaw (pd.Series): list of yaws
            altitude (float): flight altitude
            yaw (float): flight yaw
            pitch (float): flight pitch
            roll (float): flight roll
            flight_mode (FlightMode): flight mode
            error (float, optional): yaw tolerance. Defaults to 10.

        Returns:
            List[Mapping[str, int | float | None]]: List of flight lines with specifications for each line
        """

        threshold = np.median(captures_yaw)
        indexes = np.where(captures_yaw < threshold)[0]
        indexes = np.where( (np.median(captures_yaw[indexes]) - error <= captures_yaw) & (captures_yaw <= np.median(captures_yaw[indexes]) + error))[0]

        lines = FlightProcessor.compute_lines([], indexes)

        threshold = np.median(captures_yaw)
        indexes = np.where(captures_yaw > threshold)[0]

        indexes = np.where( (np.median(captures_yaw[indexes]) - error <= captures_yaw) & (captures_yaw <= np.median(captures_yaw[indexes]) + error))[0]

        lines = FlightProcessor.compute_lines(lines, indexes)
        lines.sort()

        flight_lines = [{'start' : line[0], 'end' : line[1], 'yaw' : yaw, 'pitch' : pitch, 'roll' : roll, 'alt' : altitude} for line in lines]
        
        if flight_mode in [FlightMode.NO_OVERLAP_NO_FIXED, FlightMode.OVERLAP_NO_FIXED]:
            for line in flight_lines[1::2]:
                line['yaw'] = float((yaw + 180) % 360)

        return flight_lines

    @staticmethod
    def save_metadata(data : pd.DataFrame, out_name : str) -> str:
        """Save a pd.DataFrame in a csv file

        Args:
            data (pd.DataFrame): data to save
            out_name (str): full filename where to save

        Returns:
            str: full filename where to save
        """

        out_folder = os.path.dirname(out_name)
        os.makedirs(out_folder, exist_ok = True)
        save_path = os.path.join(out_folder, out_name)

        data.to_csv(save_path)
        return save_path

    @staticmethod
    def save_flight_lines(data : List[Mapping[str, int | float | None]], out_name : str) -> str:
        """A function that saves the flight lines in a yaml file

        Args:
            data (List[Mapping[str, int  |  float  |  None]]): data
            out_name (str): full file name where to save

        Returns:
            str: full file name where to save
        """

        out_folder = os.path.dirname(out_name)
        os.makedirs(out_folder, exist_ok = True)
        save_path = os.path.join(out_folder, out_name)

        with open(save_path, 'w') as yml_file:
            data = {'flight' : data}
            yaml.dump(data, yml_file, sort_keys = False)
        
        return save_path

    @staticmethod
    def load_metadata(in_folder : str, in_name : str, sep : str = ',') -> pd.DataFrame:
        """A function that loads the metadata from a csv file

        Args:
            in_folder (str): where the csv file is
            in_name (str): its name
            sep (str, optional): separator for data. Defaults to ','.

        Returns:
            pd.DataFrame: metadata
        """

        return pd.read_csv(os.path.join(in_folder, in_name), sep = sep)

    @staticmethod
    def load_flight_lines(in_folder : str, in_name : str) -> List[Mapping[str, int | float | None]]:
        """A function that loads flight lines from a yaml file

        Args:
            in_folder (str): where the yaml file is
            in_name (str): its name

        Returns:
            List[Mapping[str, int | float | None]]: the flight lines
        """

        with open(os.path.join(in_folder, in_name), "r") as yml_file:
            try:
                flight_lines = yaml.safe_load(yml_file).get('flight', [])
            except yaml.YAMLError as exc:
                print(exc)

        return flight_lines

    @staticmethod
    def georefence_images(metadata : pd.DataFrame, partition : GeoreferencePartition, 
                          flight_lines : List[Mapping[str, int | float | None]], 
                          in_folder : str, out_folder : str, use_metadata : bool = False, **kwargs):
        """A function that georefences images from a given pd.DataFrame containing metadata and 
        the flight lines to be used

        Args:
            metadata (pd.DataFrame): captures metadata
            partition (GeoreferencePartition): which flight lines will be used
            flight_lines (List[Mapping[str, int  |  float  |  None]]): flight lines
            in_folder (str): where the images are
            out_folder (str): whre to save them
            use_metadata (bool, optional): if True we use the Source column of the metadata param to get the image paths. 
                                            Defaults to False.
        """
        
        georefence_by_uuid = GeorefenceUtils.get_georefence_by_uuid(metadata, flight_lines[slice(partition.start, 
                                                                                                 partition.end, partition.steps)])
        out_folder = os.path.join(out_folder, partition.name)

        if use_metadata:
            GeorefenceUtils.georeference_images_with_metadata(metadata, georefence_by_uuid, in_folder, out_folder, asdict(partition.profile), **kwargs)
        else:
            GeorefenceUtils.georeference_images(georefence_by_uuid, in_folder, out_folder, asdict(partition.profile), **kwargs)

    @staticmethod
    def georefence_bands(metadata : pd.DataFrame, partition : GeoreferencePartition, 
                         flight_lines : List[Mapping[str, int | float | None]], in_folder : str, 
                         out_folder : str, use_metadata : bool = False, **kwargs):
        """A function that georefences capture bands from a given pd.DataFrame containing metadata and 
        the flight lines to be used

        Args:
            metadata (pd.DataFrame): captures metadata
            partition (GeoreferencePartition): which flight lines will be used
            flight_lines (List[Mapping[str, int  |  float  |  None]]): flight lines
            in_folder (str): where the images are
            out_folder (str): whre to save them
            use_metadata (bool, optional): if True we use the Source column of the metadata param to get the capture paths. 
                                            Defaults to False.
        """

        georefence_by_uuid = GeorefenceUtils.get_georefence_by_uuid(metadata, flight_lines[slice(partition.start, partition.end, partition.steps)])
        out_folder = os.path.join(out_folder, partition.name)
        
        if use_metadata:
            GeorefenceUtils.georeference_bands_with_metadata(metadata, georefence_by_uuid, in_folder, out_folder, asdict(partition.profile), **kwargs)
        else:
            GeorefenceUtils.georeference_bands(georefence_by_uuid, in_folder, out_folder, asdict(partition.profile), **kwargs)

    @staticmethod
    def merge(metadata : pd.DataFrame, partition : MergePartition, 
              flight_lines : List[Mapping[str, int | float | None]], in_folder : str, out_folder : str, **kwargs):
        """A function that merges flight captures using the given metadata, partition and flight_lines parameters

        Args:
            metadata (pd.DataFrame): flight captures metadata
            partition (MergePartition): which flight lines will be merged
            flight_lines (List[Mapping[str, int  |  float  |  None]]): flight lines
            in_folder (str): where te captures are
            out_folder (str): where to save them
        """

        in_folder = os.path.join(in_folder, partition.name)
        out_folder = os.path.join(out_folder, f'{partition.name} - {partition.skip - 1} skip', kwargs.get('method', 'mean'))
        out_name = os.path.join(out_folder, f'{os.path.basename(os.path.dirname(in_folder))}_{kwargs.get("method", "mean")}.tif')

        os.makedirs(out_folder, exist_ok = True)
        
        raster_paths = []
        
        for line in flight_lines[slice(partition.start, partition.end, partition.steps)]:
            for uuid in metadata.iloc[line['start'] : line['end']]['ID'][::partition.skip]:
                raster_paths.append(os.path.join(in_folder, uuid))
            
        MergeUtils.merge(raster_paths = raster_paths, out_name = out_name, **kwargs)

    @staticmethod
    def resample(in_folder : str, out_folder : str, resamplings : List[Resampling]):
        """A function that resamples all the captures in the given in_folder using the resamplings settings

        Args:
            in_folder (str): Where the files to resample are
            out_folder (str): Where to save those resampled files
            resamplings (List[Resampling]): A list will all the resampling settings to be used
        """

        RasterUtils.resample(in_folder, out_folder, resamplings)


class MicaSenseProcessor(FlightProcessor):
    """This class inherits from FlightProcessor and it's used to process flight with the MicaSense sensor
    """

    @staticmethod
    def get_stacks(in_folder : str, out_folder : str, exiftool_path : str, 
                   warp_matrix : np.array, overwrite : bool = True) -> List[capture.Capture]:
        """
        Given an in_folder, an out_folder and an exiftool_path, read each capture and save
        the bands of a same capture in one file per capture.

        Args:
            in_folder (str): folder where to read each capture
            out_folder (str): folder where to write the irradiance file
            exiftool_path (str): path of the exiftool file to read metadata of each capture
            warp_matrix (np.array): a matrix to align the bands
            overwrite (bool, optional): overwrite or not. Defaults to True.

        Returns:
            List[Capture]: list of initial captures
        """
        
        os.makedirs(out_folder, exist_ok = True)

        image_set : image_set.ImageSet = imageset.ImageSet.from_directory(in_folder, exiftool_path = exiftool_path)
        captures : List[capture.Capture] = image_set.captures
        
        for idx, capture in tqdm(enumerate(captures), total = len(captures)):
            full_output_path : str = os.path.join(out_folder, str(idx) + '.tif')

            if (not os.path.exists(full_output_path) or overwrite) and (len(capture.images) == len(captures[0].images)):
                capture.compute_undistorted_radiance()
                capture.create_aligned_capture(irradiance_list = None, img_type = 'radiance', warp_matrices = warp_matrix)
                capture.save_capture_as_stack(full_output_path)
            capture.clear_image_data()

        return captures
    
    @staticmethod
    def get_thumbnails(in_folder : str, out_folder : str, exiftool_path : str, 
                       warp_matrix : np.array, overwrite : bool = True) -> List[capture.Capture]:
        """
        A function that creates thumbnail images

        Args:
            in_folder (str): where the captures are
            out_folder (str): where to save the thumbnail images
            exiftool_path (str): path of the exiftool file to read metadata of each capture
            warp_matrix (np.array): a matrix to align all bands
            overwrite (bool, optional): overwrite or not. Defaults to True.

        Returns:
            List[Capture]: list of initial captures
        """

        os.makedirs(out_folder, exist_ok = True)
        image_set : image_set.ImageSet = imageset.ImageSet.from_directory(in_folder, exiftool_path = exiftool_path)
        captures : List[capture.Capture] = image_set.captures

        for idx, capture in tqdm(enumerate(captures), total = len(captures)):
            full_output_path : str = os.path.join(out_folder, str(idx) + '.jpg')

            if (not os.path.exists(full_output_path) or overwrite) and (len(capture.images) == len(captures[0].images)):
                capture.create_aligned_capture(irradiance_list = None, img_type = 'reflectance', warp_matrices = warp_matrix)
                capture.save_capture_as_rgb(full_output_path)
            capture.clear_image_data()

        return captures
        
    @staticmethod
    def get_captures_metadata(in_folder : str) -> pd.DataFrame:
        """ A function that given a in_folder, return a pd.DataFrame with the captures metadata

        Args:
            in_folder (str): where the captures are

        Returns:
            pd.DataFrame: metadata
        """

        captures = imageset.ImageSet.from_directory(in_folder).captures
        lines : List[str] = []

        for idx, capture in tqdm(enumerate(captures), total = len(captures)):
            metadata = MicaSenseProcessor.get_capture_metadata(idx, capture)
            lines.append(list(metadata.values()))
        else:
            header = list(metadata.keys())
        
        return pd.DataFrame(columns = header, data = lines)

    @staticmethod
    def get_capture_metadata(idx : int, capture : capture.Capture) -> Mapping[str, Any]:
        """A function that reads the capture metadata and returns a dict with its values

        Args:
            idx (int): index of the capture
            capture (capture.Capture): capture

        Returns:
            Mapping[str, Any]: metadata values
        """

        def decdeg2dms(dd : float) -> Tuple[float, float, float]:
            minutes, seconds = divmod(abs(dd) * 3600, 60)
            degrees, minutes = divmod(minutes, 60)
            degrees : float = degrees if dd >= 0 else -degrees
            
            return (degrees, minutes, seconds)
        
        width, height = capture.images[0].meta.image_size()
        img : Image_micasense = capture.images[0]
        lat, lon, alt = capture.location()

        latdeg, londeg = decdeg2dms(lat)[0], decdeg2dms(lon)[0]
        latdeg, latdir = (-latdeg, 'S') if latdeg < 0 else (latdeg, 'N')
        londeg, londir = (-londeg, 'W') if londeg < 0 else (londeg, 'E')
            
            
        datestamp, timestamp = capture.utc_time().strftime("%Y-%m-%d,%H:%M:%S").split(',')
        resolution : Tuple[float, float] = capture.images[0].focal_plane_resolution_px_per_mm
        focal_length : float = capture.images[0].focal_length
        sensor_size : Tuple[float, float] = width / img.focal_plane_resolution_px_per_mm[0], height / img.focal_plane_resolution_px_per_mm[1]
            
        data = {
                'ID' : f'{idx}.tif',
                'GPSDateStamp' : datestamp,
                'GPSTimeStamp' : timestamp,
                'GPSLatitude' : lat,
                'GPSLatitudeRef' : latdir,
                'GPSLongitude' : lon,
                'GPSLongitudeRef' : londir,
                'GPSAltitude' : alt,
                'SensorX' : sensor_size[0],
                'SensorY' : sensor_size[1],
                'FocalLength' : focal_length,
                'Yaw' : (capture.images[0].dls_yaw * 180 / math.pi) % 360,
                'Pitch' : (capture.images[0].dls_pitch * 180 / math.pi) % 360,
                'Roll' : (capture.images[0].dls_roll * 180 / math.pi) % 360,
                'SolarElevation' : capture.images[0].solar_elevation,
                'ImageWidth' : width,
                'ImageHeight' : height,
                'XResolution' : resolution[1],
                'YResolution' : resolution[0],
                'ResolutionUnits' : 'mm',
            }
        
        return data

    @staticmethod
    def compute_warp_matrix(img_dir : str, exiftool_path : str, max_iterations : int = 10, 
                            match_index : int = 3, warp_mode : int = cv2.MOTION_HOMOGRAPHY, 
                            pyramid_levels : int = 1) -> np.array:
        """
        This function uses the MicaSense imageutils.align_capture() function to determine an alignment (warp) matrix of a single capture that can be applied to all images. From MicaSense: "For best alignment results it's recommended to select a capture which has features which visible in all bands. Man-made objects such as cars, roads, and buildings tend to work very well, while captures of only repeating crop rows tend to work poorly. Remember, once a good transformation has been found for flight, it can be generally be applied across all of the images." Ref: https://github.com/micasense/imageprocessing/blob/master/Alignment.ipynb

        Args:
            img_dir (str): where the capture to be aligned is located
            exiftool_path (str): path of the exiftool file to read metadata of each capture
            max_iterations (int, optional): The maximum number of solver iterations. Defaults to 50.
            match_index (int, optional): the band index to be used as the base for the rest. Defaults to 3.
            warp_mode (int, optional): warp mode. Defaults to cv2.MOTION_HOMOGRAPHY.
            pyramid_levels (int, optional): pyramid_levels. Defaults to 1.

        Returns:
            np.array: a alignment matrix
        """

        img_capture = imageset.ImageSet.from_directory(img_dir, exiftool_path = exiftool_path).captures[0]

        match_index = match_index
        warp_mode = warp_mode
        pyramid_levels = pyramid_levels
        
        warp_matrices, _ = imageutils.align_capture(img_capture,
                                                                ref_index = match_index,
                                                                max_iterations = max_iterations,
                                                                warp_mode = warp_mode,
                                                                pyramid_levels = pyramid_levels)

        return warp_matrices


class CubertProcessor:
    """This class inherits from FlightProcessor and it's used to process hyperspectral flights
    """

    @staticmethod
    def generate_flight_stacks(in_folder : str, out_folder : str, profile : Mapping[str, Any], 
                               bands : List[int] = None):
        """
        Reads files in the in_folder and generate rasters in the out_folder based on profile and bands parameters.        

        Args:
            in_folder (str): folder to read the data. We need (.cue, .hdr) files
            out_folder (str): folder to save the files
            profile (Mapping[str, Any]): Metadata of the final files
            bands (List[int], optional): if not None, bands to read and save in one file. Defaults to None.
        """

        os.makedirs(out_folder, exist_ok = True)

        file_paths : List[str] = os.listdir(in_folder)

        for index, file_path in enumerate(file_paths[::2]):
            cue_path : str = os.path.join(in_folder, file_path) if '.cue' in file_path else os.path.join(in_folder, file_paths[index * 2 + 1])
            hdr_path : str = os.path.join(in_folder, file_path) if '.hdr' in file_path else os.path.join(in_folder, file_paths[index * 2 + 1])

            img = spectral.io.envi.open(hdr_path, cue_path)
            data : np.ndarray = img.read_bands(bands = bands) if bands is not None else img.asarray()

            with rasterio.open(os.path.join(out_folder, os.path.basename(cue_path).replace('.cue', '.tif')), 'w', **profile) as dst:
                dst.write(np.moveaxis(data, (0, 1, 2), (1, 2, 0)))


@cache
def get_sensor(sensor_type : SensorType, file_paths : str, **kwargs) -> Sensor:
    """A factory function to create a Sensor objects filled with its specifications.

    Args:
        sensor_type (SensorType): sensor type
        file_paths (str): file for metadata extraction

    Returns:
        Sensor: a Sensor object
    """

    sensor : Sensor = None

    if sensor_type == SensorType.MICASENSE:
        capture = imageset.ImageSet.from_directory(os.path.dirname(file_paths), **kwargs).captures[0]
        data = MicaSenseProcessor.get_capture_metadata(0, capture)
        sensor = Sensor(data['FocalLength'], data['SensorX'], data['SensorY'], data['ImageHeight'], data['ImageWidth'], capture.num_bands, capture.band_names(), sensor_type)
    elif sensor_type == SensorType.DJI:
        data = FlightProcessor.get_capture_metadata(0, file_paths)
        count = 3
        sensor = Sensor(data['FocalLength'], data['SensorX'], data['SensorY'], data['ImageHeight'], data['ImageWidth'], count, [str(i) for i in range(count)], sensor_type)

    return sensor

@cache
def get_processor(sensor_type : SensorType) -> FlightProcessor:
    """Factory function to generate a FlightProcessor-like objects

    Args:
        sensor_type (SensorType): sensor type

    Returns:
        FlightProcessor: a FlightProcessor object
    """

    processor : FlightProcessor = None

    if sensor_type == SensorType.MICASENSE:
        processor = MicaSenseProcessor()    
    elif sensor_type == SensorType.DJI:
        processor = FlightProcessor()

    return processor
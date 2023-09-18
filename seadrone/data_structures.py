from dataclasses import dataclass
from typing import List
from pyproj import CRS
from rasterio.enums import Resampling
from fractions import Fraction

from seadrone.enums import SensorType

@dataclass
class Sensor:
    """Sensor specifications

    Args:
        focal_length (float): focal_length
        sensor_x (float): correspondence mm to pixel in width direction
        sensor_y (float): correspondence mm to pixel in height direction
        height (int): height
        width (int): width
        bands_number (int): number of bands
        band_names (List[str]): names of all bands
        name (SensorType): type of sensor
    """

    focal_length : float
    sensor_x : float
    sensor_y : float
    height : int
    width : int
    bands_number : int
    band_names : List[str]
    name : SensorType


@dataclass
class Profile:
    """Class to help rasterio to write rasters
    
    Args:
        dtype (str): data type
        count (int): number of bands
        height (int): height
        width (int): width
        nodata (int): which value represents no data
        driver (str): type of file
        crs (CRS): coordinate reference system
    """

    dtype : str
    count : int
    height : int
    width : int
    nodata : int
    driver : str = 'GTiff'
    crs : CRS = CRS.from_user_input(4326)


@dataclass
class Partition:
    """Class to represent which flight lines will be used
    
    Args:
        name (str): name of the partition
        start (int|None): index of the first line to be used
        end (int|None): index of the last line to be used
        steps (int|None): how many lines to skip
    """

    name : str
    start : int | None
    end : int | None
    steps : int | None

    @property
    def id(self):
        return f'{self.name} [{self.start} : {self.end} : {self.steps}]'


@dataclass
class MergePartition(Partition):
    """Class to represent a partition during merge process
    
    Args:
        skip (int): how many captures in a same flight line will be skipped
    """

    skip : int


@dataclass
class GeoreferencePartition(Partition):
    """Class to represent a partition during georeference process
    
    Args:
        profile (Profile): the profile data need to write a georeferenced file
    """

    profile : Profile

@dataclass
class Resampling:
    """Class to represent a resampling specification
    
    Args:
        method (Resampling): which method will be used to resample
        scale_x (int): scale in width direction
        scale_y (int): scale in height direction
        dowscale (bool): whether to downscale or upscale
    """

    method : Resampling
    scale_x : int
    scale_y : int
    dowscale : bool

    @property
    def id(self):
        return f'{self.method.name}_{"downsampling" if self.dowscale else "upsampling"}_x{self.print_scale(self.scale_x)}_y{self.print_scale(self.scale_y)}'

    def print_scale(self, scale : Fraction) -> int:
        return scale.denominator if self.dowscale else scale.numerator

    def __post_init__(self):
        self.scale_x = Fraction(1, self.scale_x) if self.dowscale else Fraction(self.scale_x, 1)
        self.scale_y = Fraction(1, self.scale_y) if self.dowscale else Fraction(self.scale_y, 1)
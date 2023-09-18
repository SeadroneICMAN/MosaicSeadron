from enum import Enum

class FlightMode(Enum):
    """Enum to represent all the possible flight modes
    """
    NO_OVERLAP_NO_FIXED : str = 'No overlap No fixed'
    NO_OVERLAP_FIXED : str = 'No overlap fixed'
    OVERLAP_NO_FIXED : str = 'Overlap No fixed'
    OVERLAP_FIXED : str = 'Overlap fixed'


class SensorType(Enum):
    """Enum to define all sensors available
    """
    MICASENSE : str = 'MicaSense'
    DJI : str = 'DJI'
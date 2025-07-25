# SeaDroneLib

MosaicSeadron is a repository that includes all the necessary scripts to georeference, mosaic, and comprehensively map consecutive overlapped captures taken by Unmanned Aerial Vehicles (UAVs, or drones) equipped with RGB, thermal, and multispectral sensors. 

More information about the methods employed in this repository can be found in the research article "Enhancing Georeferencing and Mosaicking Techniques over Water Surfaces with High-Resolution Unmanned Aerial Vehicle (UAV) Imagery", authored by [Alejandro Román](https://github.com/alrova96), [Sergio Heredia](https://github.com/Aouei), [Anna E. Windle](https://github.com/aewindle110), Antonio Tovar Sánchez, and Gabriel Navarro. The article is published in: https://www.mdpi.com/2072-4292/16/2/290. 

In accordance with the FAIR principles, all datasets (RGB, thermal, and multispectral) used in this study - except for the Lake Erie (US) dataset - can be downloaded from the following DIGITAL.CSIC repository: http://hdl.handle.net/10261/395796.

The repository includes features such as:
- Extraction of camera metadata from Micasense sensors (MicaSense RedEdge MX - 5 and 10 multispectral bands).
- Extraction of camera metadata from DJI sensors (DJI Mavic 2 Enteprise Advanced - RGB and thermal; DJI Zenmuse H20T - RGB and thermal; DJI Zenmuse L2 - RGB; DJI Mavic 3 Enterprise).
- Georeferencing of individual UAV captures.
- Mosaicking of consecutive UAV captures.
- Filtering of captures and/or mosaics.

Upcoming code updates will include adaptations for the DJI Phantom 4 Multispectral and the MicaSense RedEdge-P (both multispectral sensors).

## Folder Structure


<img src="/captures/Folder_structure.png" alt="Folder Structure" title="Folder Structure">

In general, the following folders will always be present:
- dependencies
Contains additional files required to run the code.

- micasense
Includes code for processing images captured with the MicaSense camera.

- scripts
Contains files used to execute configuration tasks, such as installing or uninstalling the MicaSense-related code.

- seadrone
Includes code for georeferencing and mosaicking.

- seadrone_usage
Contains scripts demonstrating typical use cases, such as Rrs extraction, georeferencing, and more.

## Set Up
Here we explain how to download and install the code to begin working with it.

Clone the project

```bash
  git clone https://github.com/Aouei/seadronelib
```

Go to the project directory

```bash
  cd seadronelib
```

**Important**: The environment must have python3.10 to work.

Create a virtual environment with **venv on Windows**
```bash
  python3.10 -m venv seadronelib-venv --prompt="seadronelib"
```

Activate the environment with **venv**
```bash
  seadronelib-venv\Scripts\activate
```

Create a virtual environment with **anaconda on Windows**
```bash
  conda create -n seadronelib python=3.10 -y
```

Activate the environment with **anaconda**
```bash
  conda activate seadronelib
```

Install the dependencies
```bash
  python -m pip install -r requirements.txt
  python pip install dependencies\GDAL-3.4.3-cp310-cp310-win_amd64.whl 
```

Install micasense and seadrone modules

**Important**: The virtual environment must be activated

Generation of install.sh file and install micasense and seadrone
```bash
  python -B .\scripts\package_manager_generator.py -e seadronelib-venv -p micasense,seadrone -i 1 -ri 1
```

Generation of uninstall.sh file once micasense and seadrone were installed
```bash
  python -B .\scripts\package_manager_generator.py -e seadronelib-venv -p micasense,seadrone -u 1
```

If you want to install micasense and seadrone
```bash
  \scripts\install.sh

```
If you want to uninstall micasense and seadrone
```bash
  \scripts\uninstall.sh
```
## Technical Description

The code is mainly composed of 4 files:
- raster.py: 
  - contains the code related to georeference, mosaicking, etc...
- data_structures.py: 
  - contains classes for gathering useful data needed in the processing.
- enums.py: 
  - contains enums to summarize some useful data like sensors available, etc...
- processing.py: 
  - contains the code to process a single flight in a simpler and user friendly manner.

<img src="/captures/Class_diagram.jpg" alt="Class Diagram" title="Class Diagram">

Each file (folder symbol) contains a set of classes (rectangles) with methods to solve the needs we have such as extract rrs, georeference, etc...

Each method has its own documentation explaining what it does and what each parameter it needs means.
## Usage

### Micasense processing
#### Dataset/Flight structure
In order to perform a complete processing we need our dataset to comprise the following structure:

- align_img: 
  - Contains a capture that will be used to align the rest of the dataset capture (in tif format).
- panel: 
  - Contains the captures on the panel of the dataset for further calibration if necessary (in tif format).
- raw_sky_imgs: 
  - Contains the captures of the dataset panel for further calibration if necessary (in tif format).
- raw_water_imgs: 
  - Contains the captures of the flight in question (in tif format).
- flight_lines.yaml: 
  - Contains which captures define the usable lines for georeferencing, merging, etc...

#### Usage
See: [mono_processing_jupyter/dji](/seadrone_usage/mono_processing_jupyter/dji.ipynb) and [batch_processing_jupyter/dji](/seadrone_usage/batch_processing_jupyter/dji.ipynb) files

### DJI processing
#### Dataset/Flight structure
In order to perform a complete processing we need our dataset to comprise the following structure:

- bands: 
  - Contains the captures of the flight in question (in tif format).
- main: 
  - Contains the captures of the flight in question (in jpg format).
- summary.yml: 
  - It is a file used to define flight metadata. 
  - Its main function is to determine which captures will be georeferenced and joined.

#### Usage
See: [mono_processing_jupyter/micasense](/seadrone_usage/mono_processing_jupyter/micasense.ipynb) and [batch_processing_jupyter/micasense](/seadrone_usage/batch_processing_jupyter/micasense.ipynb) files

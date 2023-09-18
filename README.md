# SeaDroneLib

MosaicSeadron is a repository that includes all the necessary scripts to georeference, mosaic, and fully map consecutive overlapped captures taken by unmanned aerial vehicles (UAVs) equipped with RGB, thermal, and multispectral sensors. More information about the methods employed in this repository can be found in "Enhancing Georeferencing and Mosaicking Techniques over Water Surfaces with High-Resolution UAV Imagery", a research article authored by [Alejandro Román](https://github.com/alrova96), [Sergio Heredia](https://github.com/Aouei), [Anna E. Windle](https://github.com/aewindle110), Antonio Tovar Sánchez, and Gabriel Navarro.

The code inside gathers features like:
- Extract camera metadata of Micasense sensors.
- Extract camera metadata of DJI sensors.
- Georeference individual UAV captures.
- Merge (mosaic) consecutive UAV captures.
- Filter captures and/or merges.

## Folder Structure


<img src="/captures/Folder_structure.png" alt="Folder Structure" title="Folder Structure">

In general there will always be the following folders:
- dependencies: 
  - contains extra files that are necessary to run the code.
- micasense: 
  - code to process the screenshots taken with the micasense camera
- scripts: 
  - files that allow to execute configuration functionalities such as installing or uninstalling the micasense code
- seadrone:
  - code for geo-referencing and mosaicking
- seadrone_usage: 
  - codes that collect typical use cases such as rrs extraction, georeferencing, ...
## Set Up
Here we explain how to download and install the code to be able to work with it.

Clone the project

```bash
  git clone https://github.com/Aouei/seadronelib
```

Go to the project directory

```bash
  cd seadronelib
```

Create a virtual environment with venv
```bash
  python -m venv seadronelib-venv --prompt="seadronelib"
```

Activate the environment
```bash
  seadronelib-venv\Scripts\activate
```

Install the dependencies
```bash
  python -m pip install -r requirements.txt
  python pip install dependencies\GDAL-3.4.3-cp310-cp310-win_amd64.whl 
```

Install micasense and seadrone modules

Important: The virtual environment must be activated

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
- water_quality.py: 
  - contains the code needed to obtain water quality products such as chlorophyll, turbidity, etc...
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
- summary.yml: 
  - It is a file used to define flight metadata. 
  - Its main function is to determine which captures will be georeferenced and joined (we will detail it).

#### Usage
[Manual](/manuals/Mosaicking_Code_Multispectral_DroneWQ.pdf)

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
Work in progress

## Examples
Work in progress

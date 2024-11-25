## czi2ashlar instructions

Installing and running czi2ashlar, a wrapper around Jeremy Muhlich's ASHLAR package (https://github.com/labsyspharm/ashlar) to interpret OHSU-style cycif multiplex image series acquired on Zeiss Axioscan scanners and inject metadata into the resulting OME-TIFF output.<br>
NOTE: tested on Linux, Mac x86 and arm64, and recently also on Windows<br>

0) Ensure your system has the development and runtime tools needed<br>
   a) ONLY ON MAC: Open a terminal window and install Apple's XCode developer tools: `xcode-select --install`<br>

1) You can skip this step if you have some other Conda installed already<br>
ON LINUX and MAC: Install Mamba (the better/faster conda, see: https://mamba.readthedocs.io/en/latest/mamba-installation.html)<br>
   a) Open Terminal window<br>
   b) `curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"`<br>
   c) `bash Mambaforge-$(uname)-$(uname -m).sh`<br>
ON WINDOWS: Install Anaconda (from: https://www.anaconda.com/download )<br>
   a) Follow the Anaconda installation instructions<br>
   b) Open Anaconda Navigator and select Environments from the left tabs<br>
   c) Click "Create" at the bottom to create a new environment called "ashlar" with Python package 3.10  (whichever the latest 3.10 version is)<br>
   d) Start the ashlar environment with the green arrow and select "Open Terminal" and continue at step 2d)<br>

2) Create a conda env and install custom ashlar, see: https://github.com/dsudar/ashlar/tree/rotcor_metadata<br>
ON MAC or LINUX:<br>
   Make sure you have a Terminal window opened<br>
   a) `conda create -y -n ashlar python=3.10`<br>
   b) `conda activate ashlar`<br>
ON ALL PLATFORMS:<br>
   c) ONLY ON MAC: `pip install pyjnius`       (needs to be done via pip upfront since there's no compatible package on condaforge)<br>
   d) `conda install -y -c conda-forge numpy scipy matplotlib networkx scikit-image=0.19 scikit-learn "tifffile>=2023.3.15" zarr pyjnius blessed openjdk`<br>
   e) ON WINDOWS or LINUX: `pip install pylibCZIrw`<br>
   ON MAC: `pip install https://pypi.scm.io/api/package/pylibczirw/pylibCZIrw-3.5.1-cp310-cp310-macosx_10_9_universal2.whl#sha256=9eb427f96cf4ae855deda50164e723e4685ad69adaf04b2831e856bd2c2144d6`<br>
   f) `pip install git+https://github.com/dsudar/ashlar.git@rotcor_metadata`<br>

3) Running the czi2ashlar app<br>
   a) cd to the directory with the czi files<br>
   b) `czi2ashlar -o outputfile.ome.tif file1 file2 file3 ....`  (you can use wildcards for the files)<br>
          NOTE 1: the OHSU naming scheme `R<cycle_number>_marker1.marker2.marker3.marker4_<other_stuff>` is required<br>
          NOTE 2: if the counterstain (e.g. DAPI) is NOT the first channel, use `-c chan# ` to indicate the channel to use for alignment, the value `-1` indicates the last channel in each cycle <br>
   c) the resulting single outputfile.ome.tif file has all the channels and rich metadata that can be read by QiTissue and other software (e.g. QuPath)<br>

4) Next time you want to run it<br>
ON MAC or LINUX: <br>
   -) open a terminal window<br>
   a) `conda activate ashlar`<br>
   b) continue with step 3<br>
ON WINDOWS: <br>
   a) Start Anaconda Navigator, start "ashlar" from the Environments list with "Open Terminal"<br>
   b) continue with step 3<br>

Note: the czi2ashlar package also contains 2 batch processing scripts which are only available on Linux (or on Mac for a)):<br>
   a) batch_czi2ashlar: a simply bash script to run czi2ashlar on multiple datasets in separate directories or combined in one directory<br>
   b) slurm_czi2ashlar: a script specifically written for the OHSU exacloud cluster which does the same as batch_czi2ashlar but runs the tasks in parallel<br>

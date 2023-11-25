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

2) Create a conda env and install custom ashlar, see: https://github.com/dsudar/ashlar/tree/rotcor_metadata<br>
   -) Open Terminal window<br>
   a) `conda create -y -n ashlar python=3.10`<br>
   b) `conda activate ashlar`<br>
   c) ONLY ON MAC: `pip install pyjnius`       (needs to be done via pip upfront since there's no compatible package on condaforge)<br>
   d) `conda install -y -c conda-forge numpy scipy matplotlib networkx scikit-image=0.19 scikit-learn "tifffile>=2023.3.15" zarr pyjnius blessed openjdk`<br>
   e) ON WINDOWS or LINUX: `pip install pylibCZIrw`<br>
   ON MAC: `pip install https://pypi.scm.io/api/package/pylibczirw/pylibCZIrw-3.5.1-cp310-cp310-macosx_10_9_universal2.whl#sha256=9eb427f96cf4ae855deda50164e723e4685ad69adaf04b2831e856bd2c2144d6`<br>
   f) `pip install git+https://github.com/dsudar/ashlar.git@rotcor_metadata`

3) Running the czi2ashlar app<br>
   a) cd to the directory with the czi files<br>
   b) `czi2ashlar -o outputfile.ome.tif file1 file2 file3 ....`  (you can use wildcards for the files, and NOTE: the OHSU naming scheme `R<cycle_number>_marker1.marker2.marker3.marker4_<other_stuff>` is required)<br>
   c) the resulting single outputfile.ome.tif file has all the channels and rich metadata that can be read by QiTissue and other software (e.g. QuPath)<br>

4) Next time you want to run it<br>
   a) `conda activate ashlar`<br>
   b) continue with step 3<br>

Note: the czi2ashlar package also contains 2 batch processing scripts:<br>
   a) batch-czi2ashlar: a simply bash scripts to run mczi2ashlar on multiple datasets in separate directories or combined in one directory<br>
   b) slurm-czi2ashlar: a script specifically written for the OHSU exacloud cluster which does the same as batch-czi2ashlar but runs the tasks in parallel<br>

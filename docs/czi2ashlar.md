## czi2ashlar instructions

Installing and running czi2ashlar, a wrapper around Jeremy Muhlich's ASHLAR package to interpret OHSU-style cycif multiplex image series acquired on Zeiss Axioscan scanners and inject metadata into the resulting OME-TIFF output.<br>

1) Install Mamba (the better/faster conda, see: https://mamba.readthedocs.io/en/latest/mamba-installation.html)<br>
   a) Open Terminal window<br>
   b) `curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"`<br>
   c) `bash Mambaforge-$(uname)-$(uname -m).sh`<br>

2) Create a conda env and install custom ashlar, see:https://github.com/dsudar/ashlar/tree/rotcor_metadata<br>
   a) `conda create -y -n ashlar python=3.10`<br>
   b) `conda activate ashlar`<br>
   c) ONLY ON MAC: `pip install pyjnius`       (needs to be done via pip upfront since there's no compatible package on condaforge)<br>
   d) `conda install -y -c conda-forge numpy scipy matplotlib networkx scikit-image=0.19 scikit-learn "tifffile>=2023.3.15" zarr pyjnius blessed`<br>
   e) ON WINDOWS or LINUX: `pip install pylibCZIrw`<br>
   <sp>   ON MAC: `pip install https://pypi.scm.io/api/package/pylibczirw/pylibCZIrw-3.5.1-cp310-cp310-macosx_12_0_universal2.whl#sha256=b2df07af9beb60c2356f888e1248353fdd3df59946603ede82405629bc06932b`<br>
   f) `pip install git+https://github.com/dsudar/ashlar.git@rotcor_metadata`

3) Running the czi2ashlar app<br>
   a) cd to the directory with the czi files<br>
   b) `czi2ashlar -o <outputfile.ome.tif> file1 file2 file3 ....`  (the OHSU naming scheme `R<cycle>_marker1.marker2.marker3.marker4_<other_stuff>` is required)<br>
   c) the resulting single outputfile.ome.tif file has all the channels and rich metadata that can be read by QiTissue and other software (e.g. QuPath)<br>

4) Next time you want to run it<br>
   a) `conda activate ashlar`<br>
   b) continue with step 3<br>


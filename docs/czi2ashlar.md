Installing and running czi2ashlar, a wrapper around Jeremy Muhlich's ASHLAR package to interpret OHSU-style cycif multiplex image series
acquired on Zeiss Axioscan scanners and inject metadata into the resulting OME-TIFF output.

On Mac:
1) Install Mamba (the better/faster conda, see: https://mamba.readthedocs.io/en/latest/mamba-installation.html)
   a) Open Terminal window
   b) curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
   c) bash Mambaforge-$(uname)-$(uname -m).sh

2) Create a conda env and install custom ashlar (https://github.com/dsudar/ashlar/tree/rotcor_metadata)
   a) conda create -y -n ashlar python=3.10
   b) conda activate ashlar
   c) ONLY ON MAC: pip install pyjnius       (needs to be done via pip upfront since there's no compatible package on condaforge)
   d) conda install -y -c conda-forge numpy scipy matplotlib networkx scikit-image=0.19 scikit-learn "tifffile>=2023.3.15" zarr pyjnius blessed
   e) ON WINDOWS or LINUX: pip install pylibCZIrw
      ON MAC with Apple Silicon (i.e. M1/M2 processor):
      pip install https://pypi.scm.io/api/package/pylibczirw/pylibCZIrw-3.5.1-cp310-cp310-macosx_12_0_arm64.whl#sha256=22bd90548e592ca9e4d606317e9e8cb669c46edb2195954c7517078a3c6e218d
      ON MAC with x86 processor
      pip install https://pypi.scm.io/api/package/pylibczirw/pylibCZIrw-3.5.1-cp310-cp310-macosx_12_0_x86_64.whl#sha256=7c924592f79e941ef990f64f7e7fa40b5580d39776e6442d21876cae4736f9de
   f) pip install git+https://github.com/dsudar/ashlar.git@rotcor_metadata

3) Run the czi2ashlar app
   a) cd to the directory with the czi files
   b) czi2ashlar -o <outputfile.ome.tif> file1 file2 file3 ....  (the OHSU naming scheme R<cycle>_marker1.marker2.marker3.marker4_<other_stuff> is required
   c) the resulting single outputfile.ome.tif file has all the channels and rich metadata that can be read by QiTissue and other software (e.g. QuPath)

4) Next time you want to run it
   a) conda activate ashlar
   b) continue with step 3


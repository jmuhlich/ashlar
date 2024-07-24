#!/usr/bin/python3

# czi2ashlar.py: converts/tiles/registers a list of CZI files generated by the OHSU workflow into a single tiled/pyramid OME-TIFF
# this wrapper around Ashlar is mostly for metadata handling for the OHSU workflow
# copyright  Damir Sudar Quantitative Imaging Systems LLC and OHSU

import argparse
import sys
import glob
import os
import tempfile
import subprocess
import re
from pylibCZIrw import czi as pyczi
from ashlar.scripts.ashlar import main as ashlar_main

def main():
    # CZI Stitching/Registration and converter to OME-TIFF
    # based on ashlar https://github.com/jmuhlich/ashlar
    # this module extract metadata from the CZI files, saves to a csv file, and calls ashlar
    # 

    # The arguments:
    #       the list of individual CZI files
    # options:
    #       -o out.ome.tif : output path of where to save the resulting OME-TIFF file and CSV metadata table
    #       -c channel : channel to use as the reference for alignment
    #       -q : stops verbose output

    # command line processing of arguments
    parser = argparse.ArgumentParser(
            description='czi2ashlar.py: convert a list of CZI files generated by the OHSU workflow into a single pyramidal OME-TIFF file',
            )
    parser.add_argument(
            "filepaths", metavar='FILE', nargs='+',
            help="Provide list of CZI files to be processed",
            )
    parser.add_argument(
            "-o", "--output", dest="output", default="ashlar_output.ome.tif", metavar='PATH',
            help="Provide a path/name for the output OME-TIFF file",
            )
    parser.add_argument(
            '-c', '--align-channel', dest='align_channel', type=int,
            default='0', metavar='CHANNEL',
            help="Reference channel number for image alignment. Numbering starts at 0.",
            )
    parser.add_argument(
            "-q", "--quiet", dest="quiet",
            help="Suppress verbose status and progress display", action="store_true", default=False,
            )
 
    args = parser.parse_args()
 
    filepaths = args.filepaths

    verbose = not args.quiet
    
    align_chan = args.align_channel
 
    # create a name/path for the csv output file
    out_path_csv = '{}.csv'.format(args.output)
 
    # the regex pattern of the single channel registered tif files
    cycif_pattern=re.compile("^R([0-9]+)_([a-zA-Z0-9-]+)\.?([a-zA-Z0-9-]+)?\.?([a-zA-Z0-9-]+)?\.?([a-zA-Z0-9-]+)?_?([a-zA-Z]?)([0-9]+)?_([a-zA-Z0-9-_()]+).czi")
 
    # csv file collects metadata extracted from filenames
    of = open(out_path_csv, "w")
    of.write("Channel,Name,Cycle,ChannelIndex,ExposureTime,ExposureTimeUnit,Fluor,AcquisitionMode,IlluminationType,ContrastMethod,ExcitationWavelength,ExcitationWavelengthUnit,EmissionWavelength,EmissionWavelengthUnit,Color\n")
 
    if verbose: print("Version: 20240723:1620")
    if verbose: print("Metadata extracted from files found:")
    if verbose: print("Cycle\tBM1\tBM2\tBM3\tBM4\tName\t\tScene")
 
    # channel counter increments while stepping through the individual czi files
    chan_count = 0
 
    # TODO add handling of TMAs and image/scene positions
    # scene_x = scene_y = 0.0
    # tma_flag = True

    for fname in filepaths:
        inbasename = os.path.basename(fname)
        if cycif_pattern.match(inbasename):
            # if regex match found, extract all the embedded metadata from filename
            match = cycif_pattern.search(inbasename)
            cycle = int(match.group(1))
            bm1 = match.group(2)
            bm2 = match.group(3)
            bm3 = match.group(4)
            bm4 = match.group(5)
            let_coor = match.group(6)
            if match.group(7): scene = int(match.group(7))
            else: scene = 0
            basename = match.group(8)
 
            exp_time = 0.0
 
            if verbose: print("%d\t%s\t%s\t%s\t%s\t%s\t%s%d" % (cycle, bm1, bm2, bm3, bm4, basename, let_coor, scene), end = ' ')
 
            if verbose: print("    Accepting: ", inbasename)
 
            with pyczi.open_czi(fname) as czidoc:
                md_dict = czidoc.metadata

            max_chan = int(md_dict['ImageDocument']['Metadata']['Information']['Image']['SizeC'])
            if align_chan > max_chan - 1 or align_chan < 0:
                dapi_chan = max_chan - 1
            else:
                dapi_chan = align_chan
 
            # create entry for a numbered DAPI channel
            exp_time = float(md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][dapi_chan]['ExposureTime']) / 1000000.0
            fluor = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][dapi_chan]['Fluor']
            acq_mode = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][dapi_chan]['AcquisitionMode']
            illum_type = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][dapi_chan]['IlluminationType']
            con_meth = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][dapi_chan]['ContrastMethod']
            color = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][dapi_chan]['Color']
            ex_wave = float(md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][dapi_chan]['ExcitationWavelength'])
            em_wave = float(md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][dapi_chan]['EmissionWavelength'])
            dapi_line = "DAPI%d,%d,%d,%f,ms,%s,%s,%s,%s,%f,nm,%f,nm,%s" % (cycle, cycle, dapi_chan, exp_time, fluor, acq_mode, illum_type, con_meth, ex_wave, em_wave, color )

            chanindex = 0
            if align_chan == 0:
                chan_line = "%d,%s\n" % (chan_count, dapi_line )
                chanindex += 1
                chan_count += 1
                of.write(chan_line)
            if bm1:
                exp_time = float(md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['ExposureTime']) / 1000000.0
                fluor = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['Fluor']
                acq_mode = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['AcquisitionMode']
                illum_type = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['IlluminationType']
                con_meth = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['ContrastMethod']
                color = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['Color']
                ex_wave = float(md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['ExcitationWavelength'])
                em_wave = float(md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['EmissionWavelength'])
                chan_line = "%d,%s,%d,%d,%f,ms,%s,%s,%s,%s,%f,nm,%f,nm,%s\n" % (chan_count, bm1, cycle, chanindex, exp_time, fluor, acq_mode, illum_type, con_meth, ex_wave, em_wave, color )
                chanindex += 1
                chan_count += 1
                of.write(chan_line)
            if align_chan == 1:
                chan_line = "%d,%s\n" % (chan_count, dapi_line )
                chanindex += 1
                chan_count += 1
                of.write(chan_line)
            if bm2:
                exp_time = float(md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['ExposureTime']) / 1000000.0
                fluor = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['Fluor']
                acq_mode = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['AcquisitionMode']
                illum_type = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['IlluminationType']
                con_meth = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['ContrastMethod']
                color = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['Color']
                ex_wave = float(md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['ExcitationWavelength'])
                em_wave = float(md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['EmissionWavelength'])
                chan_line = "%d,%s,%d,%d,%f,ms,%s,%s,%s,%s,%f,nm,%f,nm,%s\n" % (chan_count, bm2, cycle, chanindex, exp_time, fluor, acq_mode, illum_type, con_meth, ex_wave, em_wave, color )
                chanindex += 1
                chan_count += 1
                of.write(chan_line)
            if align_chan == 2:
                chan_line = "%d,%s\n" % (chan_count, dapi_line )
                chanindex += 1
                chan_count += 1
                of.write(chan_line)
            if bm3:
                exp_time = float(md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['ExposureTime']) / 1000000.0
                fluor = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['Fluor']
                acq_mode = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['AcquisitionMode']
                illum_type = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['IlluminationType']
                con_meth = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['ContrastMethod']
                color = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['Color']
                ex_wave = float(md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['ExcitationWavelength'])
                em_wave = float(md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['EmissionWavelength'])
                chan_line = "%d,%s,%d,%d,%f,ms,%s,%s,%s,%s,%f,nm,%f,nm,%s\n" % (chan_count, bm3, cycle, chanindex, exp_time, fluor, acq_mode, illum_type, con_meth, ex_wave, em_wave, color )
                chanindex += 1
                chan_count += 1
                of.write(chan_line)
            if align_chan == 3:
                chan_line = "%d,%s\n" % (chan_count, dapi_line )
                chanindex += 1
                chan_count += 1
                of.write(chan_line)
            if bm4:
                exp_time = float(md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['ExposureTime']) / 1000000.0
                fluor = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['Fluor']
                acq_mode = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['AcquisitionMode']
                illum_type = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['IlluminationType']
                con_meth = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['ContrastMethod']
                color = md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['Color']
                ex_wave = float(md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['ExcitationWavelength'])
                em_wave = float(md_dict['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel'][chanindex]['EmissionWavelength'])
                chan_line = "%d,%s,%d,%d,%f,ms,%s,%s,%s,%s,%f,nm,%f,nm,%s\n" % (chan_count, bm4, cycle, chanindex, exp_time, fluor, acq_mode, illum_type, con_meth, ex_wave, em_wave, color )
                chanindex += 1
                chan_count += 1
                of.write(chan_line)
            if align_chan == 4:
                chan_line = "%d,%s\n" % (chan_count, dapi_line )
                chanindex += 1
                chan_count += 1
                of.write(chan_line)
        else:
             if verbose: print("File \"%s\" does not match regular cycle definition - skipped. " % inbasename)
 
    of.close()

    if chan_count == 0:
        if verbose: print("No valid channel files found in \"%s\" - cannot proceed." % filepaths)
        sys.exit()
 
    if verbose: print("Done with metadata extraction")
 
    # construct the argument list
    arglist = ["-q", "--flip-y", "-o", args.output, "--metadata", out_path_csv, "-c", str(align_chan)]
    if args.quiet: arglist += ["-q"]
 
    arglist += filepaths

    # print(arglist)
    ashlar_main(arglist)
 
if __name__ == '__main__':
    main()


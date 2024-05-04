# process_raw_images.py - Description:
#  Process raw 3D-SIM images and save as .npz for reconstruction using sim3d_main.py.
# Created by Ruiming Cao on May 01, 2024
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io
import os
import glob

from absl import app
from absl import flags
from jax.config import config
import json
import numpy as np
import tifffile

from nstm import utils
from nstm import sim3d_utils
from nstm import datasets

FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", None, "input data directory.")
flags.DEFINE_string("raw_path", None, "raw 3D SIM image path. regex for multiple files of time series acquisition."
                                      "Supported formats: .tif, .czi, and .npz. The .npz file has to be generated "
                                      "by process_raw_images.py script or follow the same organization. When a "
                                      ".npz file is provided, no otf_path and meta_path are needed, and patch_json "
                                      "should not be used.")
flags.DEFINE_string("otf_path", None, "OTF path for 3D SIM. regex for multiple files.")
flags.DEFINE_string("meta_path", None, "path for metadata of 3D SIM.")
flags.DEFINE_string("save_path", None, "path to save the reconstruction results. "
                                       "This will overwrite the save path in full2patch.json file.")
flags.DEFINE_bool("fast_mode_meta", True, "assume fast mode for timestamp loading.")

flags.DEFINE_float("zoomfact", 2, "zoom factor on lateral dimensions.")
flags.DEFINE_list("coord_start", None, "the starting coordinate of the FOV in zyx (3D) or yx (2D).")
flags.DEFINE_list("patch_dim", None, "the dimension of the FOV in zyx (3D) or yx (2D).")
flags.DEFINE_integer("num_stack", 1,
                     "number of image stacks to use for the reconstruction of 3D SIM.")
flags.DEFINE_integer("starting_stack", 0, "index for the first image stack used in the "
                                          "reconstruction for 3D SIM.")

flags.DEFINE_integer("ndirs", 3, "number of directions for 3D SIM.")
flags.DEFINE_integer("nphases", 5, "number of phase shifts for 3D SIM.")
flags.DEFINE_float("background_int", 100, "background flooring intensity for each pixel.")

config.parse_flags_with_absl()


def main(unused_argv):
    coord_start = [int(s) for s in FLAGS.coord_start]
    patch_dim_zyx = [int(s) for s in FLAGS.patch_dim]

    fov_zyxshw = (coord_start[0], coord_start[1], coord_start[2],
                  patch_dim_zyx[0], patch_dim_zyx[1], patch_dim_zyx[2])
    otf_path = sorted(glob.glob(os.path.join(FLAGS.data_dir, FLAGS.otf_path)))

    d = datasets.SIM3DDataLoaderMultitime(os.path.join(FLAGS.data_dir, FLAGS.otf_path),
                                          os.path.join(FLAGS.data_dir, FLAGS.meta_path),
                                          FLAGS.zoomfact, FLAGS.ndirs, FLAGS.nphases,
                                          num_timepoint=FLAGS.num_stack,
                                          start_timepoint=FLAGS.starting_stack)
    img = d.load_3DSIM_raw(os.path.join(FLAGS.data_dir, FLAGS.raw_path), fov_zyxshw,
                           background_int=FLAGS.background_int, normalize=False)

    OTF = d.load_3DSIM_OTF(otf_path)

    # load timestamps from metadata
    timestamp_phase, _ = d.load_metadata(normalize=True, avg_phase=False, single_plane_time=FLAGS.fast_mode_meta)

    print("Saving raw images, OTF and timestamps to {}".format(FLAGS.save_path))
    np.savez(FLAGS.save_path, img=img, OTF=OTF, timestamp_phase=timestamp_phase)


if __name__ == "__main__":
    app.run(main)
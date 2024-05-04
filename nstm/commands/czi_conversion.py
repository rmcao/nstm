# czi_conversion.py - Description:
#  Convert .czi image stack into .tif image stack.
# Created by Ruiming Cao on Aug 23, 2023
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

from absl import flags
from absl import app
import numpy as np
import tifffile

from nstm.datasets import SIM3DDataLoader

FLAGS = flags.FLAGS

flags.DEFINE_string("czi_path", "", "The path to the czi file.")
flags.DEFINE_string("save_path", "", "The path to save tif file after conversion.")
flags.DEFINE_integer("ndirs", 3, "The number of orientations in the SIM data.")
flags.DEFINE_integer("nphases", 5, "The number of phases in the SIM data.")
flags.DEFINE_integer("timepoint_ind", 0, "The index of timepoint for a time-series. 0 if there's a single timepoint.")
flags.DEFINE_integer("starting_ori", 0, "The index of the starting orientation (for moving window).")
flags.DEFINE_list("zyxshw", None, "The yxhw of the FOV to be extracted. If None, the whole FOV will be extracted.")
flags.DEFINE_bool("z_first", False, "Output as z -> orientation -> phase. Default: orientation -> z -> phase.")
flags.DEFINE_integer("select_ori", None, "Select a orientation index to output. "
                                         "If None, all orientations will be output.")
flags.DEFINE_bool("mix_rot", False, "Pick different rot from different "
                  "timepoints in 3D SIM. This will artificially increase the time delay.")

def main(unused_argv):
    zyxshw = None
    if FLAGS.zyxshw:
        zyxshw = [int(s) for s in FLAGS.zyxshw]

    if FLAGS.starting_ori == 0:
        d = SIM3DDataLoader(otf_path=None, meta_path=None, zoomfact=1, ndirs=FLAGS.ndirs, nphases=FLAGS.nphases,
                            start_timepoint=FLAGS.timepoint_ind)
        img = d.load_3DSIM_raw(FLAGS.czi_path, fov_zyxshw=zyxshw, noise_std=0, background_int=0, normalize=False)
    else:
        d1 = SIM3DDataLoader(otf_path=None, meta_path=None, zoomfact=1, ndirs=FLAGS.ndirs, nphases=FLAGS.nphases,
                             start_timepoint=FLAGS.timepoint_ind)
        img1 = d1.load_3DSIM_raw(FLAGS.czi_path, fov_zyxshw=zyxshw, noise_std=0, background_int=0, normalize=False)
        d2 = SIM3DDataLoader(otf_path=None, meta_path=None, zoomfact=1, ndirs=FLAGS.ndirs, nphases=FLAGS.nphases,
                             start_timepoint=FLAGS.timepoint_ind + 1)
        img2 = d2.load_3DSIM_raw(FLAGS.czi_path, fov_zyxshw=zyxshw, noise_std=0, background_int=0, normalize=False)
        img = np.concatenate((img2[:FLAGS.starting_ori], img1[FLAGS.starting_ori:]), axis=0)

    if FLAGS.mix_rot:
        assert FLAGS.starting_ori == 0, "mix_rot only works for starting_ori=0"
        print("Mixing rotations from different timepoints")
        d2 = SIM3DDataLoader(otf_path=None, meta_path=None, zoomfact=1, ndirs=FLAGS.ndirs, nphases=FLAGS.nphases,
                             start_timepoint=FLAGS.timepoint_ind + 1)
        img2 = d2.load_3DSIM_raw(FLAGS.czi_path, fov_zyxshw=zyxshw, noise_std=0, background_int=0, normalize=False)
        d3 = SIM3DDataLoader(otf_path=None, meta_path=None, zoomfact=1, ndirs=FLAGS.ndirs, nphases=FLAGS.nphases,
                             start_timepoint=FLAGS.timepoint_ind + 2)
        img3 = d3.load_3DSIM_raw(FLAGS.czi_path, fov_zyxshw=zyxshw, noise_std=0, background_int=0, normalize=False)

        img = np.stack([img[0], img2[1], img3[2]], axis=0)

    if FLAGS.select_ori is not None:
        img = img[FLAGS.select_ori].transpose((1, 0, 2, 3))
        print("output as z -> phase for orientation {}".format(FLAGS.select_ori))
    else:
        if FLAGS.z_first:
            img = img.transpose((2, 0, 1, 3, 4))
            print("output as z -> orientation -> phase")
        else:
            img = img.transpose((0, 2, 1, 3, 4))
            print("output as orientation -> z -> phase")
    print('image shape: ', img.shape)
    img = img.reshape((-1, img.shape[-2], img.shape[-1])).astype(np.uint16)
    tifffile.imwrite(FLAGS.save_path, img)


if __name__ == "__main__":
    app.run(main)

# -*- coding: utf-8 -*-
"""Generate estimated timestamps for Zeiss SIM data where the timestamps for raw images are not recorded."""

from absl import flags
from absl import app
import numpy as np
import csv
from aicsimageio.readers.czi_reader import CziReader

FLAGS = flags.FLAGS


def define_flags():
    flags.DEFINE_string("czi_path", "", "The path to the czi file.")
    flags.DEFINE_float("exposure_time", 5, "The exposure time in ms.")
    flags.DEFINE_float("delay_rot", 300, "The delay time for each grating rotation in ms. Default: 1500ms.")
    flags.DEFINE_float("delay_phase", 20, "The delay time for each phase shift in ms. Default: 100ms.")
    flags.DEFINE_float("delay_z", 20, "The delay time for each z step in ms. Default: 100ms.")
    flags.DEFINE_bool("z_first", False, "Output as z -> orientation -> phase. Default (False): orientation -> z -> phase.")


def main(unused_argv):
    if FLAGS.z_first:
        raise NotImplementedError("z_first is not implemented yet.")

    save_path = FLAGS.czi_path[:-4] + '_timestamps.csv'

    h = CziReader(FLAGS.czi_path)
    ndirs = h.shape[2]
    nphases = h.shape[1]
    dim_zyx = h.shape[5:]
    ntimepoints = h.shape[3]

    print("ntimepoints: ", ntimepoints, ", ndirs: ", ndirs, ", nphases: ", nphases, ", dim_zyx: ", dim_zyx)

    i_phase = np.arange(nphases)
    i_zstep = np.arange(dim_zyx[0])
    i_rot = np.arange(ndirs)
    i_timepoints = np.arange(ntimepoints)

    t_phase = i_phase * (FLAGS.delay_phase + FLAGS.exposure_time)
    t_zstep = i_zstep * (np.maximum(FLAGS.delay_z, FLAGS.delay_phase) + FLAGS.exposure_time + t_phase[-1])
    t_rot = i_rot * (FLAGS.delay_rot + FLAGS.exposure_time + t_zstep[-1] + t_phase[-1])
    t_timeseries = i_timepoints * (FLAGS.delay_rot + FLAGS.exposure_time + t_rot[-1] + t_zstep[-1] + t_phase[-1])

    i_timepoints, i_rot, i_zstep, i_phase = np.meshgrid(i_timepoints, i_rot, i_zstep, i_phase, indexing='ij')
    timestamp = t_timeseries[:, None, None, None] + t_rot[None, :, None, None] + t_zstep[None, None, :, None] + \
                t_phase[None, None, None, :]

    timestamp = timestamp.flatten()
    i_timepoints = i_timepoints.flatten()
    i_rot = i_rot.flatten()
    i_zstep = i_zstep.flatten()
    i_phase = i_phase.flatten()

    with open(save_path, 'w', newline='\n') as out:
        writer = csv.DictWriter(out, fieldnames=["index", "timestamp", "timepoint", "rot", "z", "phase"])
        writer.writeheader()

        for i in range(len(timestamp)):
            writer.writerow({"index": i, "timestamp": timestamp[i], "timepoint": i_timepoints[i], "rot": i_rot[i],
                             "z": i_zstep[i], "phase": i_phase[i]})


if __name__ == "__main__":
    define_flags()
    app.run(main)

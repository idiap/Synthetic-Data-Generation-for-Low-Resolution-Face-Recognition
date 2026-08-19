#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: data_slurm_submission.py
#
import subprocess
#from . import utils
#from .utils import Sample
#from .cropper import Cropper
#from .face_extractor_3d import FaceExtractor3D
#from .generator import Generator
#from .projector import Projector
#from .databases import MultipieDatabase, Database, UTKfaceDatabase
#from .latent_edit import LatentEdit
#from synthetics.project_database import project_database
#from synthetics.custom_scoring import process_experiment

def getstatusoutput(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    out, _ = process.communicate()
    return (process.returncode, out)

def submit_job(command, job_name='job'):
    print(f"submitting {job_name}")
    status, out = getstatusoutput(command)
    jobnum=str(out[:-1]).strip().split(" ")[-1][:-1]
    if (status == 0 ):
        print(f"Job '{job_name}' is number {jobnum}")
    else:
        print (f"Error submitting Job {jobnum} {job_name}")
    return status, jobnum

def main():
    interp = ["area", "cubic"]
    resolutions = [56, 28, 14]
    
    for down_interp in interp:
        for up_interp in interp:
            for size in resolutions:
                basename = f"processed_downsample_{size}_{down_interp}_{up_interp}_webface4m"

                cmd = f"sbatch shuffle_rec.run downsample {down_interp} {up_interp} {size}"
                status, jobnum_split = submit_job(cmd, basename)
    
                cmd = f'sbatch --depend=afterany:{jobnum_split} merge_recs.run {basename}'
                status, jobnum = submit_job(cmd, f"merge_{basename}")


    print(f"Done submitting experiments")

main()

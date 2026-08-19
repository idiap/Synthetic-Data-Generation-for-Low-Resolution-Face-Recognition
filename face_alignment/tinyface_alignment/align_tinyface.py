#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: align_tinyface.py
#
import os
import sys
import argparse
from pathlib import Path
import cv2

#sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

# Repository root, resolved from this file's location so the script works from
# any checkout without editing paths.
ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT))

# Root of the TinyFace distribution. Override with the TINYFACE_ROOT environment
# variable or the --input-dir flag.
TINYFACE_ROOT = Path(os.environ.get("TINYFACE_ROOT", ROOT / "data" / "tinyface"))

DEFAULT_LST        = ROOT / "face_alignment" / "tinyface_alignment" / "Probe.lst"
DEFAULT_INPUT_DIR  = TINYFACE_ROOT / "Testing_Set" / "Probe"
DEFAULT_OUTPUT_DIR = ROOT / "face_alignment" / "tinyface_alignment" / "aligned"

def get_aligned_face_insightface(img_name, app_instance, norm_crop_fn):
    img = cv2.imread(img_name)
    #app_instance.prepare(ctx_id=0, det_size=(img.shape[1], img.shape[0]))
    
    faces = app_instance.get(img, max_num=1)
    if len(faces) == 0:
        return None
    return norm_crop_fn(img, faces[0]['kps'])


def align_images(lst_path: Path, input_dir: Path, output_dir: Path,
                 method='insightface_r50', task_id: int = 0, num_tasks: int = 1,
                 output_size: int = 112, max_input_size=None):
    all_names = [line.strip() for line in lst_path.read_text().splitlines() if line.strip()]

    # Stride-based partition: task i processes indices i, i+num_tasks, i+2*num_tasks, …
    # This distributes I/O load evenly rather than giving each worker a contiguous block.
    img_names = all_names[task_id::num_tasks]

    print(f"Task {task_id}/{num_tasks}: processing {len(img_names)} / {len(all_names)} images")
    output_dir.mkdir(parents=True, exist_ok=True)

    dfa_model = None
    use_dfa_lowres = method in ('dfa-mobilenet', 'dfa-resnet50') and (
        output_size != 112 or max_input_size is not None
    )
    if method == 'insightface_r50':
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(allowed_modules=['detection'])
        app.prepare(ctx_id=0, det_size=(32, 32))
        from insightface.utils.face_align import norm_crop
    elif method == 'mtcnn':
        from face_alignment.align import get_aligned_face
    elif method in ('dfa-mobilenet', 'dfa-resnet50'):
        import torch
        from face_alignment.dfa.dfa_align import aligner_variant_to_backbone, load_dfa_model

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dfa_model = load_dfa_model(aligner_variant_to_backbone(method), device)
        if use_dfa_lowres:
            from face_alignment.dfa.dfa_align import align_bgr_frame_lowres as _dfa_align_fn
        else:
            from face_alignment.dfa.dfa_align import align_bgr_frame as _dfa_align_fn
    else:
        raise ValueError(f"Unknown --method: {method!r}")

    failed = []
    for img_name in tqdm(img_names, desc=f"Aligning [task {task_id}]"):
        if method == 'insightface_r50':
            face = get_aligned_face_insightface(input_dir / img_name, app, norm_crop)
        elif method == 'mtcnn':
            face = get_aligned_face(str(input_dir / img_name))
        else:
            img = cv2.imread(str(input_dir / img_name))
            if img is None:
                face = None
            elif use_dfa_lowres:
                face = _dfa_align_fn(
                    dfa_model, img,
                    output_size=output_size,
                    max_input_size=max_input_size,
                )
            else:
                face = _dfa_align_fn(dfa_model, img, None)

        if face is None:
            failed.append(img_name)
        else:
            out_path = output_dir / img_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if method == 'insightface_r50':
                cv2.imwrite(str(out_path), face)
            elif method == 'mtcnn':
                face.save(str(out_path))
            elif method in ('dfa-mobilenet', 'dfa-resnet50'):
                bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_path), bgr)

    return failed, len(img_names)


def main():
    # SLURM array defaults — overridden by CLI args if provided
    slurm_task_id    = int(os.environ.get("SLURM_ARRAY_TASK_ID",    0))
    slurm_num_tasks  = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    parser = argparse.ArgumentParser(description="Align tinyface images (supports SLURM array jobs)")
    parser.add_argument("--lst",        type=Path, default=DEFAULT_LST,        help="Path to image list file")
    parser.add_argument("--input-dir",  type=Path, default=DEFAULT_INPUT_DIR,  help="Directory with source images")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for aligned output images")
    parser.add_argument("--method",     type=str,  default='insightface_r50',  help="Detection and alignment method")
    parser.add_argument("--task-id",    type=int,  default=slurm_task_id,
                        help="Index of this worker (0-based). Auto-detected from SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--num-tasks",  type=int,  default=slurm_num_tasks,
                        help="Total number of workers. Auto-detected from SLURM_ARRAY_TASK_COUNT.")
    parser.add_argument("--output-size", type=int, default=112,
                        help="Side length of the aligned output image. For dfa-* methods, "
                             "any value != 112 routes through the low-res path that warps "
                             "directly from the original frame (no 160/112 detour).")
    parser.add_argument("--max-input-size", type=int, default=None,
                        help="If set, cap the source image's longer side to this many pixels "
                             "before warping (dfa-* low-res path only). Landmark detection "
                             "still runs at the network's native resolution.")

    args = parser.parse_args()

    failed, total = align_images(
        args.lst, args.input_dir, args.output_dir,
        method=args.method, task_id=args.task_id, num_tasks=args.num_tasks,
        output_size=args.output_size, max_input_size=args.max_input_size,
    )

    print(f"\nDone: {total - len(failed)}/{total} aligned successfully")
    print(f"Failure to acquire (fta): {len(failed) / float(total) * 100:.3f}%")
    #if failed:
    #    print(f"Failed ({len(failed)}):")
    #    for name in failed:
    #        print(f"  {name}")
    print("Some failed images:")
    print('\n'.join(failed[:20]))


if __name__ == "__main__":
    main()

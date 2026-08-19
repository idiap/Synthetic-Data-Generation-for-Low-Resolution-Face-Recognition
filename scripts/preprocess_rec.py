#
# SPDX-FileCopyrightText: Copyright (c) 2022 Jiankang Deng and Jia Guo
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Script: preprocess_rec.py
# Modified from InsightFace (https://github.com/deepinsight/insightface, MIT);
# see LICENSES/MIT.txt. Changes: passthrough and esrgan modes, independent down/up interpolation, shuffle index, SLURM arrays.
#
import argparse
import multiprocessing
import os
import time

import cv2
import mxnet as mx
import numpy as np


def read_worker(args, q_in):
    path_imgidx = os.path.join(args.input, "train.idx")
    path_imgrec = os.path.join(args.input, "train.rec")
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")

    # --shuffle-index-file: pre-generated globally-shuffled index list (one rec idx per line).
    # When provided, SLURM array slicing is applied to this list instead of the natural order,
    # giving a true global shuffle across all array tasks.
    if args.shuffle_index_file and os.path.exists(args.shuffle_index_file):
        print(f"Loading pre-shuffled index file: {args.shuffle_index_file}")
        imgidx = np.loadtxt(args.shuffle_index_file, dtype=np.int64)
        print(f"  {len(imgidx)} indices loaded")
    else:
        # Try to use .lst file if provided or found
        lst_file = args.lst
        if lst_file is None:
            # Try to auto-detect lst file
            possible_lst = os.path.join(args.input, "train.lst")
            if os.path.exists(possible_lst):
                lst_file = possible_lst
                print(f"Auto-detected lst file: {lst_file}")

        if lst_file and os.path.exists(lst_file):
            # Read indices from lst file (much faster!)
            print(f"Reading indices from {lst_file}...")
            imgidx_list = []
            with open(lst_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 1:
                        imgidx_list.append(int(parts[0]))
            imgidx = np.array(imgidx_list)
            print(f"Found {len(imgidx)} images from lst file")
        else:
            # Fall back to reading metadata or scanning
            s = imgrec.read_idx(0)
            header, _ = mx.recordio.unpack(s)

            if header.flag > 0:
                # First record contains metadata, start from index 1
                print(f"Dataset has metadata: {int(header.label[0])} images")
                imgidx = np.array(range(1, int(header.label[0])))
            else:
                # No metadata, need to iterate to find all images
                print("Dataset has no metadata, scanning for images...")
                imgidx_list = [0]  # Include the first record
                idx = 1
                while True:
                    try:
                        item = imgrec.read_idx(idx)
                        if item is None:
                            break
                        imgidx_list.append(idx)
                        idx += 1
                    except:
                        break
                imgidx = np.array(imgidx_list)
                print(f"Found {len(imgidx)} images")

    # Support SLURM array jobs - process only a subset
    if args.slurm_array_id is not None and args.slurm_array_count is not None:
        total_images = len(imgidx)
        images_per_task = total_images // args.slurm_array_count
        start_idx = args.slurm_array_id * images_per_task

        # Last task handles remaining images
        if args.slurm_array_id == args.slurm_array_count - 1:
            end_idx = total_images
        else:
            end_idx = start_idx + images_per_task

        imgidx = imgidx[start_idx:end_idx]
        print(f"SLURM Task {args.slurm_array_id}/{args.slurm_array_count}: "
              f"Processing images {start_idx} to {end_idx} ({len(imgidx)} images)")

    if args.shuffle:
        rng = np.random.default_rng(args.shuffle_seed)
        rng.shuffle(imgidx)

    for idx in imgidx:
        item = imgrec.read_idx(idx)
        q_in.put(item)

    q_in.put(None)
    imgrec.close()


def process_image(img, args, rng=None):
    """Apply the configured degradation to an image.

    When args.output_size is set, the result is stored at that resolution
    (no upscale-back). When unset, the image is downsampled and then resized
    back to its original H×W (legacy behaviour for the lr_* configs).
    """
    original_size = img.shape[:2]  # (height, width)

    if args.method == 'downsample':
        interp_methods = {
            'area': cv2.INTER_AREA,
            'cubic': cv2.INTER_CUBIC,
            'linear': cv2.INTER_LINEAR,
        }
        down_interp = interp_methods[args.interp_down]
        # When output_size is set, downsample directly to it; otherwise use
        # the legacy two-step path (down to args.downsample_size, then up).
        target = args.output_size if args.output_size is not None else args.downsample_size
        img_down = cv2.resize(img, (target, target), interpolation=down_interp)
        if args.output_size is not None:
            img_processed = img_down
        else:
            up_interp = interp_methods[args.interp_up]
            img_processed = cv2.resize(
                img_down, (original_size[1], original_size[0]),
                interpolation=up_interp)

    elif args.method == 'esrgan':
        from degradations import realesrgan_degradation
        if args.output_size is None:
            raise ValueError("--method esrgan requires --output-size")
        if rng is None:
            rng = np.random.default_rng()
        img_processed = realesrgan_degradation(img, args.output_size, rng)

    elif args.method == 'zoomblur':
        try:
            import albumentations as A
            transform = A.ZoomBlur(max_factor=args.zoom_factor, p=1.0)
            img_processed = transform(image=img)['image']
        except ImportError:
            print("albumentations not installed, falling back to downsample method")
            img_down = cv2.resize(img, (args.downsample_size, args.downsample_size),
                                  interpolation=cv2.INTER_AREA)
            img_processed = cv2.resize(img_down, (original_size[1], original_size[0]),
                                       interpolation=cv2.INTER_CUBIC)
    else:
        img_processed = img

    return img_processed


def write_worker(args, q_out):
    pre_time = time.time()

    if args.input[-1] == '/':
        args.input = args.input[:-1]
    dirname = os.path.dirname(args.input)
    basename = os.path.basename(args.input)

    # Create output directory name based on processing
    if args.method == 'downsample':
        if args.output_size is not None:
            # No upscale-back: store at output_size, single interpolation only.
            suffix = f"processed_noup_{args.output_size}_{args.interp_down}"
        else:
            suffix = f"processed_{args.method}_{args.downsample_size}_{args.interp_down}_{args.interp_up}"
    elif args.method == 'esrgan':
        suffix = f"processed_noup_{args.output_size}_esrgan"
    elif args.method == 'zoomblur':
        suffix = f"processed_{args.method}_{args.zoom_factor}"
    elif args.method == 'passthrough':
        suffix = "shuffled" if args.shuffle else "passthrough"
    else:
        suffix = "processed"

    output = os.path.join(dirname, f"{suffix}_{basename}")
    os.makedirs(output, exist_ok=True)

    # For SLURM array jobs, write to separate part files
    if args.slurm_array_id is not None:
        filename_base = f"train_part_{args.slurm_array_id:04d}"
        path_imgidx = os.path.join(output, f"{filename_base}.idx")
        path_imgrec = os.path.join(output, f"{filename_base}.rec")
        path_lst = os.path.join(output, f"{filename_base}.lst")
    else:
        path_imgidx = os.path.join(output, "train.idx")
        path_imgrec = os.path.join(output, "train.rec")
        path_lst = os.path.join(output, "train.lst")

    save_record = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "w")
    lst_file = open(path_lst, 'w')

    # Per-task RNG for stochastic methods (e.g. --method esrgan). Reproducible
    # across reruns; differs per SLURM array task so all tasks don't collide.
    task_offset = args.slurm_array_id if args.slurm_array_id is not None else 0
    rng = np.random.default_rng(args.shuffle_seed + 1_000_000 * task_offset)

    more = True
    count = 0
    while more:
        deq = q_out.get()
        if deq is None:
            more = False
        else:
            if args.method == 'passthrough':
                # Copy raw JPEG bytes without decode/re-encode — no quality loss.
                header, raw_bytes = mx.recordio.unpack(deq)
                if isinstance(header.label, float):
                    label = header.label
                else:
                    label = header.label[0]
                new_header = mx.recordio.IRHeader(
                    flag=header.flag, label=label, id=header.id, id2=header.id2)
                packed = mx.recordio.pack(new_header, raw_bytes)
            else:
                # Unpack and decode image for processing
                header, img = mx.recordio.unpack_img(deq)

                # Process the image
                img_processed = process_image(img, args, rng=rng)

                # Save preview images for the first 10 on SLURM task 0 or non-SLURM run
                if count < 10 and (args.slurm_array_id is None or args.slurm_array_id == 0):
                    preview_dir = os.path.join(output, "preview")
                    os.makedirs(preview_dir, exist_ok=True)
                    if args.method == 'downsample':
                        if args.output_size is not None:
                            preview_suffix = f"_noup_{args.output_size}_{args.interp_down}"
                        else:
                            preview_suffix = f"_{args.interp_down}_{args.interp_up}_{args.downsample_size}"
                    elif args.method == 'esrgan':
                        preview_suffix = f"_noup_{args.output_size}_esrgan"
                    elif args.method == 'zoomblur':
                        preview_suffix = f"_zoom_{args.zoom_factor}"
                    else:
                        preview_suffix = f"_{args.method}"
                    cv2.imwrite(os.path.join(preview_dir, f'{count}_orig.jpg'), img)
                    cv2.imwrite(os.path.join(
                        preview_dir, f'{count}_proc{preview_suffix}.jpg'), img_processed)

                if isinstance(header.label, float):
                    label = header.label
                else:
                    label = header.label[0]

                new_header = mx.recordio.IRHeader(
                    flag=header.flag, label=label, id=header.id, id2=header.id2)
                packed = mx.recordio.pack_img(
                    new_header, img_processed, quality=args.quality, img_fmt=args.img_fmt)

            save_record.write_idx(count, packed)

            # Write to lst file: index \t label \t original_id
            lst_file.write(f"{count}\t{int(label)}\t{header.id}\n")

            count += 1
            if count % 10000 == 0:
                cur_time = time.time()
                print('save time:', cur_time - pre_time, ' count:', count)
                pre_time = cur_time

    print(f"Total images processed: {count}")
    save_record.close()
    lst_file.close()
    print(f"Saved lst file: {path_lst}")


def main(args):
    queue = multiprocessing.Queue(10240)
    read_process = multiprocessing.Process(target=read_worker, args=(args, queue))
    read_process.daemon = True
    read_process.start()
    write_process = multiprocessing.Process(target=write_worker, args=(args, queue))
    write_process.start()
    write_process.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess rec files with downsampling/upsampling')
    parser.add_argument('input', help='path to source rec directory')
    parser.add_argument('--lst', type=str, default=None,
                       help='path to .lst file (auto-detects train.lst if not provided)')
    parser.add_argument('--method', type=str, default='downsample',
                       choices=['downsample', 'zoomblur', 'passthrough', 'esrgan'],
                       help='preprocessing method. "passthrough" copies raw JPEG bytes '
                            'without re-encoding (lossless, use with --shuffle to shuffle '
                            'without quality loss). "esrgan" applies the Real-ESRGAN '
                            'second-order stochastic degradation pipeline (requires '
                            '--output-size). (default: downsample)')
    parser.add_argument('--downsample-size', type=int, default=56,
                       choices=[7, 14, 28, 56],
                       help='intermediate downsampled size; ignored when --output-size '
                            'is set with --method downsample (default: 56)')
    parser.add_argument('--output-size', type=int, default=None,
                       help='final stored image size. When set, the upscale-back step '
                            'is skipped and the rec stores images at this resolution '
                            '(produces the noup_{size}_* output naming). Required for '
                            '--method esrgan.')
    parser.add_argument('--interp-down', type=str, default='linear',
                       choices=['area', 'cubic', 'linear'],
                       help='interpolation method for downsampling (default: linear)')
    parser.add_argument('--interp-up', type=str, default='linear',
                       choices=['area', 'cubic', 'linear'],
                       help='interpolation method for upscaling after downsampling (default: linear)')
    parser.add_argument('--zoom-factor', type=float, default=1.3,
                       help='max zoom factor for ZoomBlur (default: 1.3)')
    parser.add_argument('--quality', type=int, default=100,
                       help='JPEG quality for output (default: 100)')
    parser.add_argument('--img-fmt', type=str, default='.jpg',
                       choices=['.jpg', '.png'],
                       help='output image format (default: .jpg)')
    parser.add_argument('--shuffle', action='store_true',
                       help='shuffle images during processing')
    parser.add_argument('--shuffle-seed', type=int, default=42,
                       help='random seed for --shuffle (default: 42)')
    parser.add_argument('--shuffle-index-file', type=str, default=None,
                       help='path to a pre-generated globally-shuffled index file '
                            '(one source rec index per line). When provided with SLURM '
                            'array jobs, all tasks share the same global permutation, '
                            'giving a true global shuffle. Generate with: '
                            'python scripts/generate_shuffle_index.py <rec_dir>')
    parser.add_argument('--slurm-array-id', type=int, default=None,
                       help='SLURM array task ID (auto-detected from env if not provided)')
    parser.add_argument('--slurm-array-count', type=int, default=None,
                       help='total SLURM array tasks (auto-detected from env if not provided)')

    args = parser.parse_args()

    # Auto-detect SLURM array parameters from environment
    if args.slurm_array_id is None and 'SLURM_ARRAY_TASK_ID' in os.environ:
        args.slurm_array_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    if args.slurm_array_count is None and 'SLURM_ARRAY_TASK_COUNT' in os.environ:
        args.slurm_array_count = int(os.environ['SLURM_ARRAY_TASK_COUNT'])

    # Validate args coupled to --output-size.
    if args.method == 'esrgan' and args.output_size is None:
        parser.error("--method esrgan requires --output-size")

    print(f"Processing with method: {args.method}")
    if args.method == 'downsample':
        if args.output_size is not None:
            print(f"Output size (no upscale-back): {args.output_size}, "
                  f"interp: {args.interp_down}")
        else:
            print(f"Downsample size: {args.downsample_size}, Interpolation: "
                  f"down {args.interp_down}, up {args.interp_up}")
    elif args.method == 'esrgan':
        print(f"Real-ESRGAN second-order degradation, output size: {args.output_size}")
    elif args.method == 'zoomblur':
        print(f"Zoom factor: {args.zoom_factor}")
    
    if args.slurm_array_id is not None:
        print(f"SLURM Array Mode: Task {args.slurm_array_id} of {args.slurm_array_count}")
    
    main(args)

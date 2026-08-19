#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: generate_lst_from_rec.py
#
import argparse
import os
import time

import mxnet as mx


def generate_lst(input_dir, output_lst=None):
    """Generate .lst file from existing .rec/.idx files."""
    
    path_imgidx = os.path.join(input_dir, "train.idx")
    path_imgrec = os.path.join(input_dir, "train.rec")
    
    if not os.path.exists(path_imgrec) or not os.path.exists(path_imgidx):
        print(f"Error: train.rec or train.idx not found in {input_dir}")
        return
    
    if output_lst is None:
        output_lst = os.path.join(input_dir, "train.lst")
    
    print(f"Reading from: {path_imgrec}")
    print(f"Writing to: {output_lst}")
    
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
    
    # Scan through all indices to find valid records
    print("Scanning rec file...")
    start_time = time.time()
    
    idx = 0
    valid_records = []
    
    while True:
        try:
            s = imgrec.read_idx(idx)
            if s is None:
                break
            
            # Try to unpack to get label
            try:
                header, _ = mx.recordio.unpack(s)
                
                # Extract label
                if isinstance(header.label, float) or isinstance(header.label, int):
                    label = int(header.label)
                else:
                    label = int(header.label[0])
                
                # Store: index, label, id
                valid_records.append((idx, label, header.id))
                print(f"idx: {idx}, label: {label}")

                if idx==20:
                    exit()
                
                if (idx + 1) % 10000 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Processed {idx + 1} records ({elapsed:.2f}s)")
                
            except Exception as e:
                # Skip records that can't be unpacked
                pass
            
            idx += 1
            
        except Exception as e:
            # End of file or error
            break
    
    imgrec.close()
    
    print(f"Found {len(valid_records)} valid records")
    
    # Write lst file
    print(f"Writing lst file...")
    with open(output_lst, 'w') as f:
        for idx, label, rec_id in valid_records:
            # Format: index \t label \t id
            f.write(f"{idx}\t{label}\t{rec_id}\n")
    
    elapsed = time.time() - start_time
    print(f"Done! Total time: {elapsed:.2f}s")
    print(f"Output: {output_lst}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate .lst file from existing .rec/.idx files')
    parser.add_argument('input', help='directory containing train.rec and train.idx')
    parser.add_argument('--output', type=str, default=None,
                       help='output .lst file path (default: input/train.lst)')
    
    args = parser.parse_args()
    generate_lst(args.input, args.output)

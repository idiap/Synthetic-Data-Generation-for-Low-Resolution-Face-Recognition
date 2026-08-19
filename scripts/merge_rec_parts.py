#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: merge_rec_parts.py
#
import argparse
import glob
import os
import time

import mxnet as mx


def merge_rec_parts(input_dir, output_prefix="train"):
    """Merge partial rec files created by SLURM array jobs."""
    
    # Find all partial rec files
    rec_pattern = os.path.join(input_dir, "train_part_*.rec")
    idx_pattern = os.path.join(input_dir, "train_part_*.idx")
    lst_pattern = os.path.join(input_dir, "train_part_*.lst")
    
    rec_files = sorted(glob.glob(rec_pattern))
    idx_files = sorted(glob.glob(idx_pattern))
    lst_files = sorted(glob.glob(lst_pattern))
    
    if not rec_files:
        print(f"No partial rec files found in {input_dir}")
        print(f"Looking for pattern: {rec_pattern}")
        return
    
    print(f"Found {len(rec_files)} partial rec files to merge")
    
    # Create merged output files
    output_rec = os.path.join(input_dir, f"{output_prefix}.rec")
    output_idx = os.path.join(input_dir, f"{output_prefix}.idx")
    output_lst = os.path.join(input_dir, f"{output_prefix}.lst")
    
    if os.path.exists(output_rec) or os.path.exists(output_idx):
        print(f"Warning: {output_prefix}.rec or {output_prefix}.idx already exists!")
        response = input("Overwrite? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    
    writer = mx.recordio.MXIndexedRecordIO(output_idx, output_rec, "w")
    lst_out = open(output_lst, 'w')
    
    total_count = 0
    start_time = time.time()
    
    # Merge rec files
    for rec_file, idx_file in zip(rec_files, idx_files):
        part_name = os.path.basename(rec_file)
        print(f"Processing {part_name}...")
        
        reader = mx.recordio.MXIndexedRecordIO(idx_file, rec_file, "r")
        
        # Read all records from this partial file
        idx = 0
        part_count = 0
        while True:
            try:
                record = reader.read_idx(idx)
                if record is None:
                    break
                
                # Write to merged file with new index
                writer.write_idx(total_count, record)
                
                # Also get header for lst file
                try:
                    header, _ = mx.recordio.unpack(record)
                    if isinstance(header.label, float) or isinstance(header.label, int):
                        label = int(header.label)
                    else:
                        label = int(header.label[0])
                    lst_out.write(f"{total_count}\t{label}\t{header.id}\n")
                except:
                    # If we can't unpack, write a default lst entry
                    lst_out.write(f"{total_count}\t0\t0\n")
                
                total_count += 1
                part_count += 1
                idx += 1
                
                if total_count % 10000 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Merged {total_count} records ({elapsed:.2f}s)")
            except:
                # End of this partial file
                break
        
        print(f"  {part_name}: {part_count} records")
        reader.close()
    
    writer.close()
    lst_out.close()
    
    elapsed = time.time() - start_time
    print(f"\nMerging complete!")
    print(f"Total records merged: {total_count}")
    print(f"Time elapsed: {elapsed:.2f}s")
    print(f"Output files:")
    print(f"  - {output_rec}")
    print(f"  - {output_idx}")
    print(f"  - {output_lst}")
    
    # Optionally offer to delete partial files
    print(f"\nPartial files ({len(rec_files)} sets) are still present.")
    response = 'yes' #input("Delete partial files? (yes/no): ")
    if response.lower() == 'yes':
        for rec_file, idx_file in zip(rec_files, idx_files):
            os.remove(rec_file)
            os.remove(idx_file)
        for lst_file in lst_files:
            if os.path.exists(lst_file):
                os.remove(lst_file)
        print("Partial files deleted.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge partial rec files from SLURM array jobs')
    parser.add_argument('input_dir', help='directory containing train_part_*.rec files')
    parser.add_argument('--output-prefix', type=str, default='train',
                       help='output file prefix (default: train)')
    args = parser.parse_args()
    
    merge_rec_parts(args.input_dir, args.output_prefix)

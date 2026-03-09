from natsort import natsorted
import io
import os
from PIL import Image
import pandas as pd
from glob import glob
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


# Global lock for file writing
write_lock = Lock()


def process_single_row(row_data, dump_image_dir):
    """Process single row data (image saving + JSON generation)"""
    try:
        image_item = row_data["image"]
        image_path = image_item["path"]
        image_relpath = row_data["image_relpath"]

        assert image_path == image_relpath, f"{image_path} is not the same as its realpath {image_relpath}"

        # Save image
        image = image_item["bytes"]
        image = Image.open(io.BytesIO(image))

        full_image_path = os.path.join(dump_image_dir, image_path)
        
        # Ensure image directory exists
        os.makedirs(os.path.dirname(full_image_path), exist_ok=True)
        image.save(full_image_path)

        # Organize conversation data
        convers = row_data["conversations"]

        item = {
            "id": row_data["id"],
            "image": full_image_path,
            "conversations": convers,
        }

        return json.dumps(item)
    except Exception as e:
        print(f"Error processing row {row_data.get('id', 'unknown')}: {e}")
        return None


def process_parquet_file_multithread(file_path, dump_image_dir, output_file, num_threads=48):
    """
    Process a single parquet file, using multithreading for row processing internally
    This avoids loading multiple 20GB files into memory simultaneously
    """
    print(f"\n{'='*60}")
    print(f"Processing: {file_path}")
    print(f"{'='*60}")
    
    try:
        # Read entire parquet file (20GB)
        print(f"Loading parquet file...")
        df = pd.read_parquet(file_path)
        num_rows = df.shape[0]
        print(f"Loaded {num_rows} rows from {os.path.basename(file_path)}")
        
        # Prepare all row data
        print(f"Preparing row data...")
        row_data_list = []
        for i in range(num_rows):
            row_data = {
                "image": df["image"][i],
                "image_relpath": df["image_relpath"][i],
                "conversations": df["conversations"][i].tolist(),
                "id": df["id"][i],
            }
            row_data_list.append(row_data)
        
        # Release DataFrame memory
        del df
        
        # Use thread pool to process rows in parallel
        results = []
        print(f"Processing rows with {num_threads} threads...")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_single_row, row_data, dump_image_dir): idx
                for idx, row_data in enumerate(row_data_list)
            }
            
            # Collect results (with progress bar)
            with tqdm(total=num_rows, desc=f"Processing {os.path.basename(file_path)}") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        idx = futures[future]
                        print(f"Error processing row {idx}: {e}")
                    pbar.update(1)
        
        # Batch write to file
        if results:
            print(f"Writing {len(results)} results to file...")
            with write_lock:
                with open(output_file, "a") as fout:
                    for result in results:
                        fout.write(f"{result}\n")
        
        print(f"✓ Completed {os.path.basename(file_path)}: {len(results)} items processed")
        return len(results)
        
    except Exception as e:
        print(f"✗ Error processing file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    # ============ Configuration Parameters ============
    # Number of threads per file (for parallel row processing)
    NUM_THREADS_PER_FILE = 24  # Recommended: 36-64
    
    # Number of files to process concurrently (since each file is 20GB, recommended 1-2)
    # 500GB memory / 20GB per file ≈ can process 10+ files simultaneously at most
    # But considering memory overhead during processing, it's recommended to be conservative
    MAX_CONCURRENT_FILES = 3  # Recommended: 1-3 (1=sequential file processing with parallel rows, 2-3=parallel file processing)
    # ==================================
    
    base_dir = "/public/zy/covt-collection/covt-dataset"
    dump_image_dir = "/public/zy/covt-collection/images"
    output_file = "/public/zy/covt-collection/data.json"

    # Create directory
    if not os.path.exists(dump_image_dir):
        os.makedirs(dump_image_dir)

    # Clear output file
    open(output_file, "w").close()

    # Get all parquet files
    parquet_files = natsorted(glob(f"{base_dir}/*/*.parquet"))
    print(f"\n{'='*60}")
    print(f"Found {len(parquet_files)} parquet files")
    print(f"Configuration:")
    print(f"  - Threads per file: {NUM_THREADS_PER_FILE}")
    print(f"  - Max concurrent files: {MAX_CONCURRENT_FILES}")
    print(f"  - Estimated peak memory: ~{MAX_CONCURRENT_FILES * 20}GB + overhead")
    print(f"  - Your available memory: 500GB")
    print(f"{'='*60}\n")

    # Use thread pool to process files (limit concurrency)
    total_processed = 0
    
    if MAX_CONCURRENT_FILES == 1:
        # Process files sequentially, but use multithreading within each file
        print("Mode: Sequential file processing with parallel row processing\n")
        for idx, file in enumerate(parquet_files):
            print(f"\n[{idx+1}/{len(parquet_files)}] Processing file...")
            count = process_parquet_file_multithread(
                file, dump_image_dir, output_file, NUM_THREADS_PER_FILE
            )
            total_processed += count
    else:
        # Process multiple files in parallel
        print(f"Mode: Parallel file processing (max {MAX_CONCURRENT_FILES} files at once)\n")
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FILES) as executor:
            futures = {
                executor.submit(
                    process_parquet_file_multithread, 
                    file, 
                    dump_image_dir, 
                    output_file, 
                    NUM_THREADS_PER_FILE
                ): file
                for file in parquet_files
            }
            
            # Wait for completion
            for future in as_completed(futures):
                file = futures[future]
                try:
                    count = future.result()
                    total_processed += count
                except Exception as e:
                    print(f"✗ Failed to process {file}: {e}")

    print(f"\n{'='*60}")
    print(f"🎉 Processing complete!")
    print(f"Total items processed: {total_processed}")
    print(f"{'='*60}\n")

    # Convert JSONL format to standard JSON array format
    print("Converting to standard JSON array format...")
    temp_file = output_file + ".tmp"
    
    with open(output_file, "r") as f_in, open(temp_file, "w") as f_out:
        f_out.write("[\n")
        first = True
        for line in f_in:
            if line.strip():
                if not first:
                    f_out.write(",\n")
                f_out.write("  " + line.strip())
                first = False
        f_out.write("\n]")
    
    # Replace original file with formatted version
    os.replace(temp_file, output_file)
    
    print(f"✓ Output saved to: {output_file}")
    print(f"{'='*60}\n")



if __name__ == '__main__':
    main()
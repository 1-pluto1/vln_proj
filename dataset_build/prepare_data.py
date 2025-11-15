import json
import os
import tqdm
from datasets import load_dataset

def process_parquet_to_folder_dataset(parquet_path, output_folder):
    dataset = load_dataset("parquet", data_files=parquet_path, split="train", streaming=True)
    trajectory_dict = dict()
    for row in tqdm.tqdm(dataset):
        trajectory_id = row['id']
        frame_idx = row['frame_idx'] 
        log = json.loads(row['log'])
        if trajectory_id not in trajectory_dict:
            trajectory_dict[trajectory_id] = {
                'id': trajectory_id,
                'raw_logs': log['raw_logs'],
                'preprocessed_logs': log['preprocessed_logs'],
                'instruction': log['instruction'],
                'instruction_unified': log['instruction_unified'],
                'length': len(log['preprocessed_logs']),
                'images': []
            }
            trajectory_dict[trajectory_id]['images'] = {}
            trajectory_dict[trajectory_id]['images'][frame_idx]= row['image']
        else:
            trajectory_dict[trajectory_id]['images'][frame_idx] = row['image']
            # collect all images in the this trajectory
            if len(trajectory_dict[trajectory_id]['images']) == trajectory_dict[trajectory_id]['length']:
                imgs = [trajectory_dict[trajectory_id]['images'][key] for key in sorted(trajectory_dict[trajectory_id]['images'].keys())]
                trajectory_dict[trajectory_id].pop('images', None)
                os.makedirs(os.path.join(output_folder, trajectory_id), exist_ok=True)
                for i, image in enumerate(imgs):
                    img_path = os.path.join(output_folder, trajectory_id, str(i).zfill(6) + '.jpg')
                    with open(img_path, 'wb') as f:
                        image.save(f, format='JPEG')
                with open(os.path.join(output_folder, trajectory_id, 'log.json'), 'w') as f:
                    json.dump(trajectory_dict[trajectory_id], f)
                tmp = trajectory_dict.pop(trajectory_id, None)
                if tmp is not None:
                    del tmp

if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser("Convert parquet to folder dataset (images + log.json)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--parquet_root", type=str, help="递归查找该目录下所有 .parquet")
    group.add_argument("--parquet_file", type=str, help="仅转换指定的单个 .parquet 文件")
    parser.add_argument("--output_dir", type=str, required=True, help="输出文件夹数据集目录")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.parquet_file:
        print(f"[转换] {args.parquet_file} -> {args.output_dir}")
        process_parquet_to_folder_dataset(args.parquet_file, args.output_dir)
    else:
        files = glob.glob(os.path.join(args.parquet_root, "**", "*.parquet"), recursive=True)
        if not files:
            raise RuntimeError(f"未在 {args.parquet_root} 下找到 .parquet 文件")
        for f in files:
            print(f"[转换] {f} -> {args.output_dir}")
            process_parquet_to_folder_dataset(f, args.output_dir)


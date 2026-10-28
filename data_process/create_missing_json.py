import os
import json
import copy
import random
import argparse

parser = argparse.ArgumentParser(description='Generate multimodal missing JSON.')

parser.add_argument('--clean-path', type=str, required=True, help='Path to the clean JSON file.')
parser.add_argument('--missing-video-path', type=str, required=True, help='Path to missing video placeholder.')
parser.add_argument('--missing-audio-path', type=str, required=True, help='Path to missing audio placeholder.')
parser.add_argument('--missing-type', type=str, choices=['a', 'v', 'both', 'all'], required=True, help='Missing modality type.')
parser.add_argument('--missing-rate', type=str, required=True, help='Missing rate: 0.5, 0.7, 0.9, or all')
parser.add_argument('--output-dir', type=str, default=None, help='Output directory for generated JSON.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

args = parser.parse_args()
random.seed(args.seed)

# 合法缺失率定义
valid_rates = ['0.1', '0.3', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
if args.missing_rate != 'all' and args.missing_rate not in valid_rates:
    raise ValueError("❌ missing-rate must be one of: 0.5, 0.7, 0.9, or 'all'")

# 自动设置输出目录
if args.output_dir is None:
    args.output_dir = os.path.dirname(args.clean_path)
os.makedirs(args.output_dir, exist_ok=True)

# 加载干净数据
with open(args.clean_path, 'r') as f:
    data = json.load(f)
dic_list = data['data']
total_len = len(dic_list)

for item in dic_list:
    item['missing_a'] = False
    item['missing_v'] = False

# 缺失数据生成函数
def generate_missing_json(missing_type: str, missing_rate: float, suffix: str):
    missing_audio_indices = set()
    missing_video_indices = set()

    if missing_type == 'a':
        missing_audio_indices = set(random.sample(range(total_len), int(total_len * missing_rate)))
    elif missing_type == 'v':
        missing_video_indices = set(random.sample(range(total_len), int(total_len * missing_rate)))
    elif missing_type == 'both':
        num_to_sample = int(total_len * missing_rate * 0.5)
        missing_audio_indices = set(random.sample(range(total_len), num_to_sample))
        remaining_indices = list(set(range(total_len)) - missing_audio_indices)
        missing_video_indices = set(random.sample(remaining_indices, num_to_sample))

    new_dic_list = []
    for idx, dic in enumerate(dic_list):
        new_dic = copy.deepcopy(dic)
        if idx in missing_audio_indices:
            new_dic['wav'] = os.path.join(args.missing_audio_path , '{}.wav'.format(dic.get("video_id")))
            new_dic['missing_a'] = True
            #  "wav": os.path.join(args.audio_c_path, corruption, 'severity_{}'.format(severity), '{}.wav'.format(dic.get("video_id"))),
              
        if idx in missing_video_indices:
            new_dic['video_path'] = args.missing_video_path
            new_dic['missing_v'] = True
        new_dic_list.append(new_dic)

    fname = f"missing_{suffix}_{missing_rate:.2f}.json"
    output_path = os.path.join(args.output_dir, fname)
    with open(output_path, 'w') as f:
        json.dump({"data": new_dic_list}, f, indent=1)
    print(f"Saved: {output_path}")

# 主逻辑
rates_to_use = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9] if args.missing_rate == 'all' else [float(args.missing_rate)]
types_to_use = ['a', 'v', 'both'] if args.missing_type == 'all' else [args.missing_type]

for r in rates_to_use:
    for t in types_to_use:
        generate_missing_json(t, r, suffix=t)

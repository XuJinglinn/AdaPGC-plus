import os
import json
import random
import argparse
from copy import deepcopy
from typing import List, Dict, Any, Optional


def resolve_output_root(clean_path: str, output_dir: Optional[str]) -> str:
    if output_dir is not None:
        return output_dir
    clean_dir = os.path.dirname(os.path.abspath(clean_path))
    if os.path.basename(clean_dir) == "clean":
        return os.path.dirname(clean_dir)
    return clean_dir


def load_clean_list(clean_path: str) -> List[Dict[str, Any]]:
    with open(clean_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "data" in data and isinstance(data["data"], list), "clean json must have top-level key: data(list)"
    return data["data"]


def dump_json(dic_list: List[Dict[str, Any]], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"data": dic_list}, f, indent=4, ensure_ascii=False)


def parse_video_corruption_list(arg: List[str]) -> List[str]:
    default_list = [
        "gaussian_noise",
        "motion_blur",
        "frost",
        "snow",
        "contrast",
        "defocus_blur",
        "glass_blur",
        "impulse_noise",
        "zoom_blur",
        "brightness",
        "jpeg_compression",
        "pixelate",
        "shot_noise",
        'fog',
        "elastic_transform",
        # "spatter",
        # "speckle_noise",
    ]
    if len(arg) == 1 and arg[0].lower() == "all":
        return default_list
    return arg


def parse_audio_corruption_list(arg: List[str]) -> List[str]:
    default_list = [
        "missing",
        "gaussian_noise",
        "crowd",
        "rain",
        "thunder",
        "traffic",
        "wind",
    ]
    if len(arg) == 1 and arg[0].lower() == "all":
        return default_list
    return arg


def main():
    parser = argparse.ArgumentParser(description="Create JSONs for: audio corrupted + video corrupted (both modalities)")
    parser.add_argument("--clean-path", type=str, required=True, help="Path to clean JSON (e.g., .../clean/severity_0.json)")

    parser.add_argument("--audio-c-path", type=str, required=True, help="Root path of audio corruptions (e.g., .../audio_test-C)")
    parser.add_argument("--video-c-path", type=str, required=True, help="Root path of video corruptions (e.g., .../image_mulframe_test-C)")

    parser.add_argument("--audio-corruption", nargs="+", default=["all"], help="Audio corruption types or 'all'")
    parser.add_argument("--video-corruption", nargs="+", default=["all"], help="Video corruption types or 'all'")

    parser.add_argument("--severity", nargs="+", default=["all"], help="Severity list like: 1 2 3 4 5 or 'all'")

    parser.add_argument("--seed", type=int, default=2026, help="Random seed (only affects shuffle)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle list before saving")
    parser.add_argument("--output-dir", type=str, default=None, help="Output root dir. If not set, derived from clean-path location")

    args = parser.parse_args()
    random.seed(args.seed)

    # 1) load clean
    clean_list = load_clean_list(args.clean_path)

    # 2) corruption lists
    audio_corruptions = parse_audio_corruption_list(args.audio_corruption)
    video_corruptions = parse_video_corruption_list(args.video_corruption)

    # 3) severity
    if len(args.severity) == 1 and str(args.severity[0]).lower() == "all":
        severity_list = [1, 2, 3, 4, 5]
    else:
        severity_list = [int(x) for x in args.severity]

    # 4) output root
    out_root = resolve_output_root(args.clean_path, args.output_dir)

    # 5) generate
    for a_cor in audio_corruptions:
        for v_cor in video_corruptions:
            for sev in severity_list:
                dic_list = []
                for dic in clean_list:
                    new_dic = deepcopy(dic)
                    video_id = new_dic["video_id"]

                    # audio -> corrupted wav
                    new_dic["wav"] = os.path.join(
                        args.audio_c_path,
                        a_cor,
                        "severity_{}".format(sev),
                        "{}.wav".format(video_id)
                    )

                    # video -> corrupted folder
                    new_dic["video_path"] = os.path.join(
                        args.video_c_path,
                        v_cor,
                        "severity_{}".format(sev),
                        ""
                    )

                    dic_list.append(new_dic)

                if args.shuffle:
                    random.shuffle(dic_list)

                # severity 不作为目录，只写到文件名中
                save_dir = os.path.join(out_root, "both_corrupted", "audio_{}".format(a_cor)+'_'+ "video_{}".format(v_cor))
                save_path = os.path.join(save_dir, "severity_{}.json".format(sev))
                dump_json(dic_list, save_path)

                print("[OK] audio_corruption={}, video_corruption={}, severity={} -> {}".format(
                    a_cor, v_cor, sev, save_path
                ))


if __name__ == "__main__":
    main()

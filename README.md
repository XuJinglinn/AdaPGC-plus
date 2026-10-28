<div align="center">

# AdaPGC-plus

Calibrating Multimodal Test-Time Adaptation under<br>Reliability and Availability Modality Shifts


[Setup](#setup) &nbsp; · &nbsp; [Missing-Modality Data](#missing-modality-data) &nbsp; · &nbsp; [Evaluation](#evaluation)


</div>

## Setup

Please refer to [AdaPGC](https://github.com/XuJinglinn/AdaPGC) for environment setup.

```bash
git clone https://github.com/XuJinglinn/AdaPGC-plus.git
cd AdaPGC-plus
```

## Missing-Modality Data

We construct **Audio Missing**, **Visual Missing**, and **Both Missing** at rates **0.5, 0.6, 0.7, 0.8, and 0.9**. In Both Missing, half of the selected samples lose audio and the other half lose video; no sample loses both modalities.

Prepare the clean data and JSON files following [READ](https://github.com/XLearning-SCU/2024-ICLR-READ).

Run the following commands in Bash from the repository root. The example uses Kinetics50; replace the paths with your local paths. Reuse the original audio and frame directories: the loader zeros missing modalities according to the JSON flags, so no additional media files are needed.

```bash
for rate in 0.5 0.6 0.7 0.8 0.9; do
  python data_process/create_missing_json.py \
    --clean-path 'json_csv_files/ks50/clean/severity_0.json' \
    --missing-audio-path '/data/ks50/audio' \
    --missing-video-path '/data/ks50/frames' \
    --missing-type all \
    --missing-rate "$rate" \
    --output-dir 'json_csv_files/ks50/missing' \
    --seed 42
done
```

This generates **15 JSON files** under `json_csv_files/ks50/missing/`, named `missing_a_<rate>.json`, `missing_v_<rate>.json`, and `missing_both_<rate>.json`, with rates formatted as `0.50` through `0.90`.

For VGGSound, use `json_csv_files/vgg/clean/severity_0.json`, set the output directory to `json_csv_files/vgg/missing`, and replace the audio and frame paths with those of VGGSound.

## Evaluation

Update the local paths and GPU IDs in the corresponding script, then run:

```bash
# Kinetics50
bash script/ks50.sh

# VGGSound
bash script/vgg.sh
```

Each script runs audio corruption, video corruption, and missing-modality evaluation. To evaluate only missing modalities, run only the block with `--corruption-modality missing` in the corresponding script. Logs are saved under `exp_logs/`.

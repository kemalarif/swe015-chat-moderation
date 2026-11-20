"""Model weights live on the Hugging Face Hub, not in the git repository.

    python -m src.hub upload      # (author only) push artifacts/ to the Hub model repo
    ensure_artifacts()            # called by the app: download artifacts/ if they are missing
"""
import sys

from .config import ARTIFACTS, HF_REPO

FILES = ["thresholds.json", "tfidf_lr.pkl", "w2v_ffn.pt", "w2v_cnn.pt", "berturk-offenseval/*",
         "onnx/berturk_int8.onnx", "README.md"]
REQUIRED = ["thresholds.json", "tfidf_lr.pkl", "w2v_ffn.pt", "w2v_cnn.pt",
            "berturk-offenseval/config.json", "berturk-offenseval/model.safetensors",
            "onnx/berturk_int8.onnx"]


def missing():
    return [f for f in REQUIRED if not (ARTIFACTS / f).exists()]


def ensure_artifacts():
    if not missing() or HF_REPO.startswith("TODO"):
        return
    from huggingface_hub import snapshot_download
    print(f"Downloading model weights from https://huggingface.co/{HF_REPO} ...", flush=True)
    snapshot_download(HF_REPO, local_dir=str(ARTIFACTS), allow_patterns=FILES)
    if missing():
        raise FileNotFoundError(f"Still missing after download: {missing()}")


def upload():
    from huggingface_hub import create_repo, upload_folder
    create_repo(HF_REPO, repo_type="model", exist_ok=True)
    upload_folder(folder_path=str(ARTIFACTS), repo_id=HF_REPO, allow_patterns=FILES,
                  commit_message="Upload fine-tuned models")
    print(f"uploaded -> https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    if sys.argv[1:] == ["upload"]:
        upload()
    else:
        ensure_artifacts()

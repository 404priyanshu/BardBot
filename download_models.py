"""Download the models BardBot needs into data/models, once.

The other scripts load models with local_files_only=True, so they work
offline after this script has run.

    python download_models.py              # embedding model + local LLM
    python download_models.py --reranker   # also the optional reranker
"""
import argparse

from huggingface_hub import snapshot_download

CACHE_DIR = "data/models"

MODELS = {
    "BAAI/bge-small-en-v1.5": "embedding model (search), about 130 MB",
    "microsoft/Phi-4-mini-instruct": "local LLM (writes answers), about 7.7 GB",
}
OPTIONAL_RERANKER = {
    "BAAI/bge-reranker-base": "optional reranker (--rerank), about 1.1 GB",
}
# Only the files that sentence-transformers and transformers need.
# This skips duplicate weight formats (ONNX, GGUF, PyTorch .bin).
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]

parser = argparse.ArgumentParser(description="Download BardBot's models.")
parser.add_argument("--reranker", action="store_true", help="Also download the optional reranker.")
args = parser.parse_args()

models = dict(MODELS)
if args.reranker:
    models.update(OPTIONAL_RERANKER)

for repo_id, purpose in models.items():
    print(f"Downloading {repo_id} - {purpose}")
    path = snapshot_download(repo_id, cache_dir=CACHE_DIR, allow_patterns=ALLOW_PATTERNS)
    print("  Saved to:", path)

print("All models are ready in", CACHE_DIR)

import sys

from backend.math_backend import get_backend
from benchmark import evaluate
from dataset.parse_dataset import EmberDataset
from training.helpers import load_model

run_dir = sys.argv[1]
backend = get_backend()
model, mean, inv_std, cfg = load_model(f"{run_dir}/model.npz", backend)

for split in ("train", "val", "test"):
    m = evaluate(
        model, backend, EmberDataset.load(split), mean, inv_std, cfg.batch_size, cfg.clip
    )
    print(
        f"{split:5}  acc {m['accuracy']:.4f}  P {m['precision']:.4f}  "
        f"R {m['recall']:.4f}  F1 {m['f1']:.4f}  n={m['n']}"
    )

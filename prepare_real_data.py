"""Template preprocessor for adapting a real dataset to this BeltRec reproduction.

Expected logical fields per interaction:
    user_id, item_id, env_id, timestamp

Expected item-side arrays (one row per item):
    modality_visual.npy
    modality_text.npy
    modality_review.npy
    modality_supplier.npy
    modality_context.npy
    exposure.npy
    taxonomy.npy

The final directory must also contain graph.npz and metadata.json; see README.md and
`generate_synthetic.py` for exact formats.

This file intentionally does not fabricate the WFT-CB dataset because the manuscript says
it is available only on reasonable request and the dataset itself was not supplied.
"""

from pathlib import Path
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(
        "Created output directory. Copy/produce your precomputed item features and interaction "
        "records following README.md. Use generate_synthetic.py as an executable reference."
    )


if __name__ == "__main__":
    main()

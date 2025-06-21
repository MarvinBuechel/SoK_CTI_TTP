from const import MODELS, ExpDataset
from itertools import product
import json
import sys
from grid import *


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage: python gen_json.config.py (single|multi) OUTFILE [0]")
        sys.exit(-1)

    problem_type = sys.argv[1]
    if problem_type != "single" and problem_type != "multi":
        print("usage: python gen_json.config.py (single|multi) OUTFILE [0]")
        sys.exit(-1)

    outfile = sys.argv[2]
    c_id = 0 if len(sys.argv) <= 3 else int(sys.argv[3])

    if problem_type == "single":
        datasets = [
            ExpDataset.BOSCH_TECHNIQUES_SL.value,
            ExpDataset.TRAM_TECHNIQUES_SL.value,
        ]
    else:
        datasets = [
            ExpDataset.BOSCH_TECHNIQUES_10.value,
            ExpDataset.BOSCH_TECHNIQUES_25.value,
            ExpDataset.BOSCH_TECHNIQUES_50.value,
            # ExpDataset.BOSCH_TECHNIQUES_53.value,
            ExpDataset.BOSCH_TECHNIQUES.value,
            ExpDataset.TRAM_TECHNIQUES_10.value,
            ExpDataset.TRAM_TECHNIQUES_25.value,
            ExpDataset.TRAM_TECHNIQUES.value,
            # ExpDataset.TRAM_AUGMENTED_ARTIFICIAL.value,
            # ExpDataset.TRAM_AUGMENTED_OOD.value,
            # ExpDataset.TRAM_AUGMENTED_OOD_REBALANCED.value,
            # ExpDataset.BOSCH_GROUPS.value,
            # ExpDataset.BOSCH_SOFTWARE.value,
            # ExpDataset.BOSCH_TACTICS.value,
            # ExpDataset.BOSCH_ALL.value
        ]

    combos = list(
        product(
            freeze_layers,
            models,
            datasets,
            learning_rate,
            pos_weights,
            batch_size,
            epochs,
            end_factors
        )
    )
    config = []
    for fl, mdl, d, lr, pw, bs, e, ef in combos:
        config.append(
            {
                "id": c_id,
                "model_name": mdl,
                "dataset_name": d,
                "model_params": {
                    "epochs": e,
                    "batch_size": bs,
                    "freeze_layers": fl,
                    "learning_rate": lr,
                    "pos_weight": pw,
                    "end_factor": ef
                },
            }
        )
        c_id += 1
    with open(outfile, "w") as f:
        json.dump(config, f, indent=4)

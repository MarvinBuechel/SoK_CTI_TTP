#!/bin/bash

FOLDERS=("/classification/local" "/classification/fine_tuned/")

for FOLDER in "${FOLDERS[@]}"; do
    [ -d "$FOLDER" ] && [ "$(ls -A "$FOLDER")" ] && echo "$FOLDER ok" || { echo "$FOLDER does not exist"; exit 1; }
done

FILES=("/classification/datasets/bosch_test.json" \
    "/classification/datasets/bosch_train.json" \
    "/classification/datasets/tram_train.json" \
    "/classification/datasets/tram_test.json" \
    "/classification/datasets/tram_train_augmented_artificial.json" \
    "/classification/datasets/tram_train_augmented_ood.json" \
    "/classification/datasets/nvidia-bosch-test-embeddings.pickle" \
    "/classification/datasets/nvidia-bosch-train-embeddings.pickle" \
    "/classification/datasets/nvidia-tram-test-embeddings.pickle" \
    "/classification/datasets/nvidia-tram-train-embeddings.pickle" \
    "/classification/datasets/mitre_embeddings.pickle")

for FILE in "${FILES[@]}"; do
[ -f "$FILE" ] && [ -s "$FILE" ] && \
    echo "$FILE ok" || \
    { echo "$FILE missing"; exit 1; }
done

# here showing the baseline model makes sense, as TRAM2 is already trained on the TRAM dataset
python -m test_labeled --remove-dupl-models --show-baseline fine_tuned/tram_swipe table6_tram.csv $DEVICE
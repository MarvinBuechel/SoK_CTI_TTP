#!/bin/bash

# here showing the baseline model makes sense, as TRAM2 is already trained on the TRAM dataset
python -m test_labeled --remove-dupl-models --show-baseline fine_tuned/tram_swipe table6_tram.csv $DEVICE
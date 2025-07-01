# Named Entity Recognition (NER) Experiments
This directory contains the experiments of the Named Entity Recognition pipeline for the *"SoK: Automated TTP Extraction from CTI Reports – Are We There Yet?"* paper. We provide the following experiments:

## Ablation study (Table 5)
The aim is to find out to what extent the Named Entity Recognition pipeline performs if we remove given components.

We use the `ablation.sh` script to perform this experiment. This script calls the following three scripts in order:
1. `create_pipelines.sh` creates all NER pipelines used in the ablation study.
2. `run_pipelines.sh` runs all TRAM2 and Bosch AnnoCTR documents through all NER pipelines.
3. `evaluate_pipelines.sh` runs the evaluations against the ground truth.

The final results are stored in the `results` directory. Here, we store the full evaluation reports. The numbers present in the paper are the Average values from the Classification Weighted output report. These can be obtained using the following command line script:

```bash
for file in *.txt; do
  match=$(grep -m2 'Average' "$file" | tail -n1)
  [ -n "$match" ] && echo "$file: $match"
done
```

## Top-k experiment (Figure 4)
The aim is to find out to what extent the Named Entity Recognition pipeline performs if we vary the number of target Techniques.

We use the `open_set_scenario.sh` script to perform this experiment, which creates an NER pipeline, runs the Bosch AnnoCTR dataset through the pipeline, and evaluates the output for different number of target techniques.

The final results are stored in the `results/OpenSet` directory. Here, we store the full evaluation reports. The numbers present in the paper are the Average values from the Classification Weighted output report. These can be obtained using the following command line script:

```bash
for file in *.txt; do
  match=$(grep -m2 'Average' "$file" | tail -n1)
  [ -n "$match" ] && echo "$file: $match"
done
```
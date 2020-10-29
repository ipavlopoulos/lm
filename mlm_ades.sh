#!/bin/bash
ades=("ADEs/anaphylactic_995.0.csv", "generalised_localised_skin_eruption_6930.csv", "osteoporosis_73300.csv", "unspecified_ade_999.9.csv")
repetitions=1
vocab_size=20000
min_word_freq=2

for ade in "${ades[@]}"
do
  echo "INFO: Working with $ade, please wait ..."
  python run_mlm.py \
  --method counts \
  --repetitions $repetitions \
  --vocab_size $vocab_size \
  --min_word_freq $min_word_freq \
  --dataset ADEs/hypotension_458.29.csv \
  --report_type Radiology > "$ade".slm.log &
done
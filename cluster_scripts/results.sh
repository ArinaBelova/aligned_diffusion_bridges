#!/bin/bash

# Define the directory containing .out files
DIRECTORY="output_logs"

OUTPUT_FILE="./eval.log"

# Clear the output file before appending new results
> "$OUTPUT_FILE"
# 1557612
for i in {1620545..1620548}; do
    file="$DIRECTORY/${i}_te-protein.out"
    
    # Check if file exists
    if [[ -f "$file" ]]; then
        echo "Processing file: $file" >> "$OUTPUT_FILE"
    
    # Extract required lines
        inference_on=$(grep -m1 -E "^Run evaluation for.$*" "$file")
        init_h=$(grep -m1 -E "^init H in fbb.*$" "$file")
        init_k=$(grep -m1 -E "^init K fbb.*$" "$file")
        rmsd_10=$(grep -m1 -E "^RMSD \(10\):.*$" "$file")
        rmsd_less=$(grep -m1 -E "^RMSD \(% <\).*$" "$file")

        # Append the extracted lines to eval.log
        [[ -n $inference_on ]] && echo "$inference_on" >> "$OUTPUT_FILE"
        [[ -n $init_h ]] && echo "$init_h" >> "$OUTPUT_FILE"
        [[ -n $init_k ]] && echo "$init_k" >> "$OUTPUT_FILE"
        [[ -n $rmsd_10 ]] && echo "$rmsd_10" >> "$OUTPUT_FILE"
        [[ -n $rmsd_less ]] && echo "$rmsd_less" >> "$OUTPUT_FILE"
        echo "-----------------------------------" >> "$OUTPUT_FILE"
    else
        echo "File not found: $file" >> "$OUTPUT_FILE"
    fi
done

echo "Extraction completed. Results saved in $OUTPUT_FILE"
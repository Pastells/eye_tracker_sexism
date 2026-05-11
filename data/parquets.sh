#!/bin/bash

# Create destination folder
mkdir -p "all_parquets"

# Define source folders
folders=(
    "Data export - sexisme_20_1"
    "Data export - sexisme_20_2"
    "Data export - sexisme_20_3"
    "Data export - sexisme_20_4"
)

# Loop over each folder and each parquet file
for folder in "${folders[@]}"; do
    # Extract the suffix number (1, 2, 3, 4)
    suffix="${folder##*_}"

    for parquet_file in "$folder"/*.parquet; do
        # Extract base filename e.g. "sexisme_20 Recording5.parquet"
        basename_file=$(basename "$parquet_file")

        # Extract recording part e.g. "Recording5"
        recording=$(echo "$basename_file" | grep -oP 'Recording\d+')

        # Build new filename e.g. "sexisme_20_Recording5_participant1.parquet"
        new_name="sexisme_20_${recording}_${suffix}.parquet"

        cp "$parquet_file" "all_parquets/$new_name"
        echo "Copied: $parquet_file -> all_parquets/$new_name"
    done
done

echo "Done. All parquet files are in 'all_parquets/'."

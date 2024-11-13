#!/bin/bash

selected_cancers=$1

# Check if the selected cancers are provided
if [ -z "$selected_cancers" ]; then
    echo "Please provide the selected cancers as an argument."
    exit 1
fi

# Combine selected cancers by _
cancers=$(echo $selected_cancers | tr ' ' '_')
echo $cancers

# CSV file path, appended with cancers
csv_file="./data/annotations/${cancers}/empty_txts.csv"
echo $csv_file

# Check if the CSV file exists
if [ ! -f "$csv_file" ]; then
    echo "CSV file not found!"
    exit 1
fi

# Base directory for PDFs and output
base_dir="./data/annotations"
output_base_dir="./data/annotations/${cancers}/output"

# Ensure the output base directory exists
mkdir -p "$output_base_dir"

# Read and process each value from the first and second columns
while IFS=, read -r first_column cancer_subdir; do
    echo "Processing: $first_column in subdir $cancer_subdir"

    # Adjust the base_dir with the cancer subdirectory specified in the CSV
    adjusted_base_dir="${base_dir}/${cancer_subdir}"

    # Find the PDF file that includes the first_column in the file name
    pdf_file=$(find "$adjusted_base_dir" -type f -name "*${first_column}*.pdf")

    if [ -z "$pdf_file" ]; then
        echo "No PDF file found for $first_column in $adjusted_base_dir"
        continue
    fi

    echo "Found PDF: $pdf_file"

    # Get the base name of the PDF file (without the directory path)
    pdf_filename=$(basename "$pdf_file")

    # Create a temporary directory for the PDF file's images
    temp_dir="${output_base_dir}/${pdf_filename}_images"
    mkdir -p "$temp_dir"

    # Convert the PDF to images using pdftoppm
    pdftoppm "$pdf_file" "${temp_dir}/page"

    echo "PDF converted to images in $temp_dir"

    # Initialize a variable to store the concatenated text
    combined_text=""

    # Iterate through the generated images and use Tesseract to read text
    for image in "$temp_dir"/page*.ppm; do
        text=$(tesseract "$image" -)
        combined_text+="$text\n"
    done

    # Save the combined text to a file in the same directory as the PDF
    output_text_file="${adjusted_base_dir}/${pdf_filename%.pdf}.txt"
    echo -e "$combined_text" > "$output_text_file"

    echo "Text extracted and saved to $output_text_file"

done < "$csv_file"

# Remove the output directory after processing all files
rm -rf "$output_base_dir"
echo "Output directory $output_base_dir deleted."

# create new file tesseract_conversion_success.txt
touch "./data/annotations/${cancers}/tesseract_conversion_success.txt"

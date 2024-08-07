#!/bin/bash

# Hardcoded CSV file path
csv_file="./data/annotations/empty_txts.csv"

# Base directory for PDFs and output
base_dir="./data/annotations"
output_base_dir="./data/annotations/output"

# Check if the file exists
if [ ! -f "$csv_file" ]; then
    echo "CSV file not found!"
    exit 1
fi

# Ensure the output base directory exists
mkdir -p "$output_base_dir"

# Read and process each value from the first column
while IFS=, read -r first_column _; do
    echo "Processing: $first_column"

    # Find the PDF file that includes the first_column in the file name
    pdf_file=$(find "$base_dir" -type f -name "*${first_column}*.pdf")

    if [ -z "$pdf_file" ]; then
        echo "No PDF file found for $first_column"
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

    # Save the combined text to a file in the annotations directory
    output_text_file="${base_dir}/${pdf_filename%.pdf}.txt"
    echo -e "$combined_text" > "$output_text_file"

    echo "Text extracted and saved to $output_text_file"

done < "$csv_file"

# Remove the output directory after processing all files
rm -rf "$output_base_dir"
echo "Output directory $output_base_dir deleted."

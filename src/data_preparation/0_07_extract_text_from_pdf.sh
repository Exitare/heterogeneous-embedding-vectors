#!/bin/bash

# Iterate through all PDF files in data/annotations and its subfolders
for file in data/annotations/**/*.pdf; do
    # Check if the file exists
    if [ ! -f "$file" ]; then
        echo "File $file does not exist. Skipping."
        continue
    fi

    # Define output files within the same subfolder as the original PDF
    output_file="$(dirname "$file")/$(basename "$file" .pdf).txt"
    temp_file="$(dirname "$file")/$(basename "$file" .pdf)_temp.txt"

    # Extract text from PDF and suppress error messages
    if pdftotext "$file" "$temp_file" 2>/dev/null; then
        # Convert text to UTF-8 and remove invalid characters
        if iconv -f utf-8 -t utf-8//IGNORE "$temp_file" > "$output_file"; then
            echo "Successfully extracted and formatted text from $file to $output_file"
            rm "$temp_file"  # Remove temporary file after successful conversion
        else
            echo "Failed to convert $temp_file to UTF-8"
        fi
    else
        echo "Failed to extract text from $file"
    fi
done

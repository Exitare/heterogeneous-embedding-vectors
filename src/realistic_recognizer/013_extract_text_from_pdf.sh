#!/bin/bash

# Iterate through data/annotations/*.pdf
for file in data/annotations/*.pdf; do
    # Check if the file exists
    if [ ! -f "$file" ]; then
        echo "File $file does not exist. Skipping."
        continue
    fi

    # Extract text from PDF and suppress error messages
    output_file="data/annotations/$(basename "$file" .pdf).txt"
    temp_file="data/annotations/$(basename "$file" .pdf)_temp.txt"

    if pdftotext "$file" "$temp_file" 2>/dev/null; then
        # Convert text to UTF-8 and remove invalid characters
        if iconv -f utf-8 -t utf-8//IGNORE "$temp_file" > "$output_file"; then
            echo "Successfully extracted and formatted text from $file to $output_file"

            # Remove nonsensical conversions using sed
            sed -i '' '/III IIIIII1!IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII11Ill/d' "$output_file"
            sed -i '' '/III \^\^IIIlI\^I\^\^IIlHIlIIlI\^\^IIlIHI\^ItIlfIIIIIIIIIItiiIiiiiiiiII/d' "$output_file"
            sed -i '' '/III 11111IIIIIIIIIIIIIIIIIII11111111111IIIIII111111IIIIIIll/d' "$output_file"
            sed -i '' '/\/ca 6 - 3/d' "$output_file"
            sed -i '' '/C'"'"' \^ \^ rl '"'"'vul r\^ I '"'"'13 SItti; y\^,,\^,\^'"'"' \^ a!It c 73\.9/d' "$output_file"
            sed -i '' '/Y\/Alit/d' "$output_file"

            rm "$temp_file"  # Remove temporary file after successful conversion
        else
            echo "Failed to convert $temp_file to UTF-8"
        fi
    else
        echo "Failed to extract text from $file"
    fi
done

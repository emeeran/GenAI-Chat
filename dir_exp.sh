#!/bin/bash

# Prompt user for the directory path
read -p "Enter the directory path: " DIRECTORY

# Check if the provided directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory $DIRECTORY does not exist."
    exit 1
fi

# Prompt user for the output file name
read -p "Enter the output markdown file name (e.g., output.md): " OUTPUT_FILE

# Prompt user for file extensions
read -p "Enter file extensions to include (comma-separated, e.g., txt,md,py): " EXTENSIONS

# Convert comma-separated extensions to an array
IFS=',' read -r -a ext_array <<< "$EXTENSIONS"

# Create or clear the output markdown file
> "$OUTPUT_FILE"

# Function to append content to the markdown file
append_content() {
    local filename="$1"
    local filepath="$2"
    local extension="${filename##*.}"

    # Check if the file is a Python (.py) file
    if [ "$extension" == "py" ]; then
        echo "### $filename" >> "$OUTPUT_FILE" # Subtitle for Python files
    else
        echo "## $filename" >> "$OUTPUT_FILE"   # Header for other files
    fi

    echo '```' >> "$OUTPUT_FILE"               # Code block for the content
    cat "$filepath" >> "$OUTPUT_FILE"          # Append the content of the file
    echo '```' >> "$OUTPUT_FILE"               # Close the code block
    echo "" >> "$OUTPUT_FILE"                   # Add a newline for readability
}

# Loop through the contents of the directory based on specified extensions
for ext in "${ext_array[@]}"; do
    for FILE in "$DIRECTORY"/*."$ext"; do
        if [ -f "$FILE" ]; then # Ensure it is a file
            append_content "$(basename "$FILE")" "$FILE"
        fi
    done
done

echo "Contents of files with specified extensions in $DIRECTORY have been exported to $OUTPUT_FILE."

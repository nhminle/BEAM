import os

# Specify the folder containing the CSV files
folder_path = "/Users/alishasrivastava/BEAM-scripts/BEAM/results/direct_probe/Llama-3.3-70B-Instruct/ne_zero_shot"

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv") and "Meta-" in filename:
        # Create the new filename by replacing "Meta-" with an empty string
        new_filename = filename.replace("Meta-", "")

        # Get full paths
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")

import os
import re

# Define the hardcoded folder path
folder_path = "/Users/alishasrivastava/BEAM-scripts/BEAM/results/direct_probe/Llama-3.1-405b/non_ne_zero_shot"  # Change this to your actual folder path

# Iterate through files in the folder
for filename in os.listdir(folder_path):
    if re.search(r"_0s_NE_data\.csv$", filename):
        new_filename = re.sub(r"_0s_NE_data", "_zero-shot", filename)
        # Create full paths
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{filename}' to '{new_filename}'")
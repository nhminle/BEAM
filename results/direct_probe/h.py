import os

# Hardcoded path to the main 'analyses' directory
BASE_DIR = "/Users/alishasrivastava/BEAM-scripts/BEAM/results/direct_probe/EuroLLM-9B-Instruct"  # Change this to your actual path

# Iterate through all directories inside the base 'analyses' folder
for folder in os.listdir(BASE_DIR):
    folder_path = os.path.join(BASE_DIR, folder)

    # Ensure it's a directory
    if os.path.isdir(folder_path):
        new_folder_path = os.path.join(folder_path, "analyses")

        # Create the 'analyses' subfolder if it doesn't exist
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
            print(f"Created: {new_folder_path}")
        else:
            print(f"Already exists: {new_folder_path}")

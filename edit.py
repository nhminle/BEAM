import os

# Path to the direct_probe directory
direct_probe_path = "/Users/alishasrivastava/BEAM-scripts/BEAM/results/prefix_probe"

# Check if the directory exists
if not os.path.exists(direct_probe_path):
    print(f"The directory {direct_probe_path} does not exist.")
    exit()

# Iterate through each folder inside direct_probe
for folder in os.listdir(direct_probe_path):
    folder_path = os.path.join(direct_probe_path, folder)

    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Path for the instructions.md file
        instructions_file = os.path.join(folder_path, "instructions.md")
        
        # Create the empty instructions.md file
        with open(instructions_file, 'w') as file:
            pass  # Create an empty file

        print(f"Created: {instructions_file}")

print("All folders have been processed.")

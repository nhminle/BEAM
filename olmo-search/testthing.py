import os

# Step 1: Build a file dictionary from the project folder using os.walk.
def build_file_dict(project_folder):
    file_dict = {}
    for dirpath, _, filenames in os.walk(project_folder):
        for fname in filenames:
            file_dict[fname] = os.path.join(dirpath, fname)
    return file_dict

# Step 2: Define a function that, given the target filename (e.g., "1984_non_NE"),
# extracts the primary part ("1984") and then searches the file dictionary for a file
# whose name starts with that primary part.
def find_matching_file(target, file_dict, secondary_suffix="non_NE"):
    # Remove extension if present in target (in case your CSV includes extensions)
    target_base = os.path.splitext(target)[0]  # e.g., "1984_non_NE"
    
    # Extract primary part:
    # If the target ends with "_non_NE", remove that part.
    suffix_with_sep = "_" + secondary_suffix
    if target_base.endswith(suffix_with_sep):
        primary_target = target_base[:-len(suffix_with_sep)]
    else:
        # Alternatively, fallback to the first token before an underscore
        primary_target = target_base.split("_")[0]
    
    # Search the file dictionary: assume that the real file's name also starts with the same primary part.
    for fname, full_path in file_dict.items():
        base_fname = os.path.splitext(fname)[0]
        # Here we assume that if the real file's base name, when split by underscore,
        # has the same first token as our primary, then it's a match.
        if base_fname.split("_", 1)[0] == primary_target:
            return full_path
    # If no match is found, return None.
    return None

# Example usage:

# Let's assume your project folder is the current directory.
project_folder = "."

# Build the file dictionary (mapping from filename to full path)
files = build_file_dict(project_folder)

# Now, suppose your main CSV's filename column for a record is "1984_non_NE"
target_filename = "1984_non_NE"

# Find the matching complex file:
matching_file_path = find_matching_file(target_filename, files)

if matching_file_path:
    print(f"Found matching file: {matching_file_path}")
else:
    print("Matching file not found.")

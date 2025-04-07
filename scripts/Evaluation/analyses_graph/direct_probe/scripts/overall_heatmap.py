import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# 📂 List of CSV files (modify this to include all your files)
CSV_FILES = [
    "scripts/Evaluation/analyses_graph/direct_probe/og/os_ne_aggregate_data_grouped.csv",
    "scripts/Evaluation/analyses_graph/direct_probe/og/os_masked_aggregate_data_grouped.csv",
    "scripts/Evaluation/analyses_graph/direct_probe/og/os_non_ne_aggregate_data_grouped.csv",
    "scripts/Evaluation/analyses_graph/direct_probe/og/zs_ne_aggregate_data_grouped.csv",
    "scripts/Evaluation/analyses_graph/direct_probe/og/zs_masked_aggregate_data_grouped.csv",
    "scripts/Evaluation/analyses_graph/direct_probe/og/zs_non_ne_aggregate_data_grouped.csv"
]

# 📌 Define names for the Y-axis (modify this to match the CSV files)
Y_AXIS_NAMES = [
    "One Shot",
    "OS Masked",
    "OS Non NE",
    "Zero Shot",
    "ZS Masked",
    "ZS Non NE"
]

# 📂 Output directory (where the heatmap and CSV will be saved)
OUTPUT_DIR = "scripts/Evaluation/analyses_graph/direct_probe/og"

def create_aggregate_heatmap(csv_files, y_axis_names, output_dir):
    """
    Generates a heatmap of aggregate accuracy across CSV files.
    X-axis: English, Translations, Cross-lingual
    Y-axis: Custom names for each CSV
    """
    df_list = []
    
    for file, name in zip(csv_files, y_axis_names):
        try:
            # Load CSV
            df = pd.read_csv(file)

            # Ensure necessary columns exist
            if not {"English", "Translations", "Cross-lingual"}.issubset(df.columns):
                print(f"Skipping {file}: Missing required columns.")
                continue

            # Select only the required accuracy columns
            acc_data = df[["English", "Translations", "Cross-lingual"]].mean().to_frame().T
            acc_data.index = [name]  # Set row index as dataset name

            df_list.append(acc_data)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    if not df_list:
        print("No valid data found. Skipping heatmap.")
        return

    # Combine all data into a single DataFrame
    agg_df = pd.concat(df_list)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save aggregated data to CSV
    csv_output_path = os.path.join(output_dir, "aggregate_heatmap_data.csv")
    agg_df.to_csv(csv_output_path, index=True)
    print(f"Saved CSV: {csv_output_path}")

    # Custom colormap
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_bupu',
        ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
        N=256
    )

    # Plot heatmap
    plt.figure(figsize=(10, len(y_axis_names) * 0.5 + 4))
    sns.heatmap(
        agg_df, annot=True, fmt=".1f", cmap=custom_cmap,
        vmin=0, vmax=100,
        cbar_kws={"label": "Accuracy (%)"}, annot_kws={"size": 12}
    )

    plt.xlabel("Language Group", fontsize=14)
    plt.ylabel("Dataset", fontsize=14)
    plt.title("Direct Probe: Experiments by Aggregate Accuracy", fontsize=16)

    # Save heatmap
    heatmap_output_path = os.path.join(output_dir, "direct_probe_aggregate_accuracy_heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_output_path, dpi=300)
    plt.close()
    print(f"Saved heatmap: {heatmap_output_path}")

# Run the script
create_aggregate_heatmap(CSV_FILES, Y_AXIS_NAMES, OUTPUT_DIR)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re

def find_latest_csv(directory="."):
    """Finds the most recently modified CSV file in the given directory."""
    csv_files = glob.glob(os.path.join(directory, "run_summary_*.csv"))
    if not csv_files:
        print("Error: No 'run_summary_*.csv' files found in the current directory.")
        return None
    latest_file = max(csv_files, key=os.path.getmtime)
    print(f"Using data from: {latest_file}")
    return latest_file

def clean_time_ms(time_val):
    """Converts time string (e.g., '123.45 ms') to float, handles '–' or errors."""
    if pd.isna(time_val) or time_val == "–":
        return None
    try:
        # Remove " ms" if present and convert
        return float(str(time_val).lower().replace(" ms", ""))
    except ValueError:
        return None

def extract_shape_dims(shape_str):
    """Extracts H, W, C from shape string like '13x13x256' or '8x13x256 (est. local 8x)' """
    if pd.isna(shape_str) or shape_str == "–":
        return None, None, None
    match = re.match(r"(\d+)x(\d+)x(\d+)", str(shape_str))
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None


def plot_performance_by_np(df, version_prefix, title_suffix, output_filename):
    """Plots Time_ms vs. NP for versions matching the prefix."""
    plt.figure(figsize=(10, 6))
    subset_df = df[df['Version'].str.startswith(version_prefix) & df['Success']]
    
    if subset_df.empty:
        print(f"No successful runs found for {version_prefix} to plot performance.")
        plt.close() # Close empty figure
        return

    sns.lineplot(data=subset_df, x='NP', y='Time_ms', hue='Version', marker='o', errorbar=None)
    sns.scatterplot(data=subset_df, x='NP', y='Time_ms', hue='Version', legend=False, s=100)

    plt.title(f'Performance: Time vs. Number of Processes ({title_suffix})')
    plt.xlabel('Number of Processes (NP)')
    plt.ylabel('Time (ms)')
    plt.grid(True, which="both", ls="--")
    plt.xticks(subset_df['NP'].unique()) # Ensure all NP values are shown as ticks
    plt.legend(title='Version')
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved: {output_filename}")
    plt.close()


def plot_correctness_comparison(df, output_filename):
    """Compares 'First_Values' for single-process successful runs of V1, V3, V4."""
    plt.figure(figsize=(12, 7))
    
    # Filter for successful NP=1 runs of specific versions
    np1_df = df[
        (df['NP'] == 1) &
        df['Success'] &
        (df['Version'].isin(['V1 Serial', 'V3 CUDA', 'V4 MPI+CUDA']))
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    if np1_df.empty:
        print("No successful NP=1 runs found for V1, V3, V4 to compare correctness.")
        plt.close()
        return

    # For simplicity in plotting, we'll just use the 'First_Values' string.
    # A more rigorous comparison would parse these into numbers.
    # We'll create a categorical plot showing the first few values.
    
    # Limit the string length for display
    np1_df['Display_Values'] = np1_df['First_Values'].str.slice(0, 40) + '...' # Truncate for readability

    # Create a bar for each version, with 'Display_Values' as categories (not ideal but illustrative)
    # A textual table might be better for direct comparison of these strings.
    
    unique_values = np1_df[['Version', 'Display_Values']].drop_duplicates()

    if unique_values.empty:
        print("No data to plot for correctness comparison.")
        plt.close()
        return
        
    value_counts = unique_values['Display_Values'].value_counts()
    
    if not value_counts.empty:
        sns.barplot(x=value_counts.index, y=value_counts.values, palette="viridis")
        plt.title('Comparison of First Output Values (NP=1, Successful Runs)')
        plt.xlabel('Sample Output Values (Truncated)')
        plt.ylabel('Number of Versions with this Output')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_filename)
        print(f"Saved: {output_filename}")
    else:
        print("Could not generate correctness comparison plot (no differing values or data).")
    plt.close()


def plot_success_summary(df, output_filename):
    """Plots a bar chart summarizing success/failure counts per Version and NP."""
    plt.figure(figsize=(14, 8))
    
    # Create a combined 'Version_NP' column for easier grouping
    df_copy = df.copy()
    df_copy['Version_NP'] = df_copy['Version'] + " (NP=" + df_copy['NP'].astype(str) + ")"
    
    # Count successes and failures
    # A 'Success' boolean column is expected (True/False)
    success_counts = df_copy.groupby('Version_NP')['Success'].value_counts().unstack(fill_value=0)
    
    if success_counts.empty:
        print("No data to plot for success summary.")
        plt.close()
        return

    success_counts.plot(kind='bar', stacked=True, colormap='RdYlGn', edgecolor='black')
    
    plt.title('Run Success Summary')
    plt.xlabel('Version and Number of Processes')
    plt.ylabel('Number of Runs')
    plt.xticks(rotation=45, ha="right")
    plt.legend(title='Outcome (True=Success)')
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved: {output_filename}")
    plt.close()

def main():
    sns.set_theme(style="whitegrid")
    csv_file = find_latest_csv()
    if not csv_file:
        return

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file '{csv_file}' is empty.")
        return
    except Exception as e:
        print(f"Error reading CSV file '{csv_file}': {e}")
        return

    if df.empty:
        print("CSV file is empty. No data to plot.")
        return

    # Data Cleaning
    df['Time_ms'] = df['Time_ms'].apply(clean_time_ms)
    df['Success'] = df['Success'].astype(bool) # Ensure 'Success' is boolean

    # --- Generate Plots ---
    
    # 1. Performance of V2 versions
    plot_performance_by_np(df, 'V2 ', 'V2 MPI Only', 'performance_v2.png')

    # 2. Performance of V4
    plot_performance_by_np(df, 'V4 MPI+CUDA', 'V4 MPI+CUDA', 'performance_v4.png')
    
    # 3. Overall Success/Failure Summary
    plot_success_summary(df, 'success_summary.png')

    # 4. Correctness - Comparison of V1, V3 (NP=1), V4 (NP=1) sample output
    # This is a qualitative comparison.
    plot_correctness_comparison(df, 'correctness_sample_comparison.png')
    
    # 5. Bar chart of execution times for NP=1 runs (V1, V3, V4)
    plt.figure(figsize=(10, 6))
    np1_times_df = df[
        (df['NP'] == 1) & 
        df['Success'] & 
        df['Version'].isin(['V1 Serial', 'V3 CUDA', 'V4 MPI+CUDA']) &
        df['Time_ms'].notna() # Only include rows where Time_ms is not None/NaN
    ]
    if not np1_times_df.empty:
        sns.barplot(data=np1_times_df, x='Version', y='Time_ms', palette="coolwarm", hue='Version', dodge=False)
        plt.title('Execution Time for Single Process Runs (Successful)')
        plt.xlabel('Version')
        plt.ylabel('Time (ms)')
        plt.tight_layout()
        plt.savefig('performance_np1_comparison.png')
        print(f"Saved: performance_np1_comparison.png")
    else:
        print("No successful NP=1 runs with valid times found for V1, V3, V4 to compare performance.")
    plt.close()

    print("\nGraph generation complete. Check for .png files in the current directory.")

if __name__ == "__main__":
    main()
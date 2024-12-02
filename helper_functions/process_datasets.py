import os
import shutil
import pandas as pd

def process_online_dataset(txt_file_path, source_folder, destination_folder):
    """
    Processes a text file where each line has the format `<file name> <corresponding move>`,
    copies the file from the source folder to the destination folder, and renames it by
    prepending 'onlinedata_' to the file name.

    Args:
        txt_file_path (str): Path to the text file with file names and moves.
        source_folder (str): Path to the source folder containing the files.
        destination_folder (str): Path to the destination folder to copy the files to.

    Returns:
        None
    """
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    with open(txt_file_path, 'r') as file:
        for line in file:
            # Extract the file name from the line
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 1:
                print(f"Skipped empty or malformed line: {line}")
                continue
            original_file_name = parts[0]

            # Construct source and destination paths
            source_path = os.path.join(source_folder, original_file_name)
            new_file_name = f"onlinedata_{original_file_name}"
            destination_path = os.path.join(destination_folder, new_file_name)

            if os.path.exists(source_path):
                # Copy and rename the file
                shutil.copy(source_path, destination_path)
                print(f"Copied and renamed: {original_file_name} -> {new_file_name}")
            else:
                print(f"File not found: {source_path}")

# Example usage
txt_file_path = "online_dataset/training_tags.txt"
source_folder = "online_dataset/extracted_move_boxes"
destination_folder = "processed_online_dataset"

process_online_dataset(txt_file_path, source_folder, destination_folder)


def process_prof_dataset(csv_path, source_folder, destination_folder, output_txt_path):
    """
    Processes a CSV file and copies corresponding files to a new folder with renamed files.
    Additionally, creates a .txt file logging the renamed files and their corresponding moves.

    Args:
        csv_path (str): Path to the CSV file.
        source_folder (str): Path to the folder containing the source files.
        destination_folder (str): Path to the folder where files will be copied and renamed.
        output_txt_path (str): Path to the .txt file to be created.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Filter rows where 'move_state' is 'valid'
    valid_moves = df[df['move_state'] == 'valid']
    
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Open the output text file for writing
    with open(output_txt_path, 'w') as output_file:
        for _, row in valid_moves.iterrows():
            # Determine the original file name using the 'id' column
            file_id = int(row['id'])
            png_filename = f"{file_id}.png"
            jpe_filename = f"{file_id}.jpe"
            
            # Check if the file exists as .png or .jpe
            if os.path.exists(os.path.join(source_folder, png_filename)):
                original_filename = png_filename
            elif os.path.exists(os.path.join(source_folder, jpe_filename)):
                original_filename = jpe_filename
            else:
                print(f"File not found for ID {file_id}: neither {png_filename} nor {jpe_filename}")
                continue
            
            # Full path of the original file
            original_filepath = os.path.join(source_folder, original_filename)
            
            # Create the new file name and its path
            new_filename = f"profdata_{original_filename}"
            new_filepath = os.path.join(destination_folder, new_filename)
            
            # Copy the file to the destination with the new name
            shutil.copy(original_filepath, new_filepath)
            
            # Write the new file name and the corresponding move to the .txt file
            corresponding_move = row['prediction']
            output_file.write(f"{new_filename} {corresponding_move}\n")
            
            print(f"Processed: {original_filename} -> {new_filename}")
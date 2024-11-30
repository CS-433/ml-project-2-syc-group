import os

def rename_chess_files(folder_path):
    """
    Renames chess files in a folder based on the specified format:
    <game id>_<colour>_<move number>.png

    Args:
        folder_path (str): Path to the folder containing the files to rename.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            try:
                # Split the filename to extract relevant parts
                parts = filename.split("_")
                if len(parts) >= 4: # Ensure the file name matches the expected format
                    game_id = parts[0] # First 3 digits represent game id
                    colour = parts[3] # Determine the colour
                    move_number = parts[2] # Move number
                    
                    # Construct new filename
                    new_name = f"dataset1_{game_id}_{colour}_{move_number}.png"
                    
                    # Rename the file
                    old_path = os.path.join(folder_path, filename)
                    new_path = os.path.join(folder_path, new_name)
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_name}")
                else:
                    print(f"Skipped: {filename} (unexpected format)")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import chess.pgn 
import argparse 
import os 


def get_box_positions(img)-> list[tuple[int, int, int, int]]: 
    """
    Returns the positions of the boxes in the image (x, y, width, height) 
    Hardcoded based on the CHESS.COM scoresheet layout. 
    
    :param img: np.ndarray, the image
    :return: list of tuples, the positions of the boxes in the image
    """
    
    x , y , w , h = 161, 321, 186, 42 
    
    rects = [] 
    for i in range(0, 25): 
        new_x = x 
        new_y = y + i * (h-1)
        rects.append((new_x, new_y, w, h))
        
    for i in range(0, 25): 
        new_x = x + w 
        new_y = y + i * (h-1)
        rects.append((new_x, new_y, w, h))
        
    # sort the rects 
    rects = sorted(rects, key=lambda x: (x[1],x[0])) 

    right_rects = [] 
    for i in range(0, 25): 
        new_x = x + 2 * w + 50  
        new_y = y + i * (h-1)
        right_rects.append((new_x, new_y, w, h))

    for i in range(0, 25): 
        new_x = x + 3 * w + 57
        new_y = y + i * (h-1)
        right_rects.append((new_x, new_y, w, h))
    right_rects = sorted(right_rects, key=lambda x: (x[0],x[1])) 

    rects.extend(right_rects) 
    return rects 

def extract_moves_from_image(img_path:str)->list[np.ndarray]: 
    """
    Extracts the moves from the image at the given path. 
    
    :param img_path: str, the path to the image
    :return: list of np.ndarrays, the extracted moves
    """
    img = cv2.imread(img_path) 

    rects = get_box_positions(img) 
    moves = [] 
    for i, rect in enumerate(rects): 
        x, y, w, h = rect 
        crop = img[y:y+h, x:x+w]
        moves.append(crop) 
    return moves

   
def extract_game_moves_san_frompgn(game_idx, destination_path, pgns_dataset_path):
    """
    Extract the moves of a specified game from a PGN file and save them in SAN notation.
    
    :param game_idx: int, the index of the game in the PGN file
    :param destination_path: str, the directory to save the moves
    :param pgns_dataset_path: str, the path to the PGN file
    """
    with open(pgns_dataset_path) as pgn_file:
        # Locate the desired game
        for _ in range(int(game_idx) + 1):  # Skip to the game index
            game = chess.pgn.read_game(pgn_file)
        
        if not game:
            print(f"Game {game_idx} not found.")
            return
        
        # Get moves in SAN notation
        board = game.board()
        moves_san = []
        for move in game.mainline_moves():
            moves_san.append(board.san(move))
            board.push(move)
        
        # Create directory for saving moves
        dir_path = os.path.join(destination_path, f"game{game_idx}")
        os.makedirs(dir_path, exist_ok=True)

        # Write SAN moves to a file
        san_str = ""
        for i, move in enumerate(moves_san):
            if i % 2 == 0:
                san_str += f"\n{i // 2 + 1}."
            san_str += f" {move}"
        
        with open(os.path.join(dir_path, "moves_san.txt"), "w") as f:
            f.write(san_str.strip())
    
    print(f"SAN moves saved for game {game_idx} in {dir_path}")
            
                
    
    
    
    
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--game_id", type=str, required=True)
    parser.add_argument("--destination_path", type=str, required=True)
    parser.add_argument("--pgns_dataset_path", type=str, required=True)
    args = parser.parse_args()
    
    extract_game_moves_san_frompgn(args.game_id, args.destination_path, args.pgns_dataset_path)
    
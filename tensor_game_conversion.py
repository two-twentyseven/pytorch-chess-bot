import torch
import chess

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    tensor = torch.zeros(855, dtype=torch.float32)  # 832 (piece encoding) + 23 (metadata: turn, castling, halfmove, en_passant, repetition)

    # Piece type → index in one-hot (0–12)
    piece_to_index = {
        None: 0,  # empty square
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6,
    }

    # Fill piece encoding (64 × 13 = 832)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        base = square * 13  # start index in tensor for this square

        if piece is None:
            tensor[base] = 1.0  # empty square
        else:
            offset = piece_to_index[piece.piece_type]
            if piece.color == chess.BLACK:
                offset += 6  # shift to black piece indices (7–12)
            tensor[base + offset] = 1.0

    # Turn (white = 1.0, black = 0.0)
    tensor[832] = float(board.turn)

    # Castling rights
    tensor[833] = float(board.has_kingside_castling_rights(chess.WHITE))
    tensor[834] = float(board.has_queenside_castling_rights(chess.WHITE))
    tensor[835] = float(board.has_kingside_castling_rights(chess.BLACK))
    tensor[836] = float(board.has_queenside_castling_rights(chess.BLACK))

    # Fifty-move rule counter (clipped and normalized)
    tensor[837] = min(board.halfmove_clock, 100) / 100.0

    # En passant square (16 one-hot encoding)
    ep_square = board.ep_square
    if ep_square is not None:
        file = chess.square_file(ep_square)
        rank = chess.square_rank(ep_square)
        if rank == 5:  # white captures possible
            index = file
        elif rank == 2:  # black captures possible
            index = 8 + file
        else:
            index = None
        if index is not None:
            tensor[838 + index] = 1.0

    # Repetition danger flag (1 if position already seen twice, else 0)
    tensor[854] = float(board.is_repetition(2))

    return tensor

def build_move_index():
    index_to_move = {}
    index = 0

    # Step 1 & 2: Generic "piece movement" using queen + knight move logic
    for from_square in chess.SQUARES:
        from_rank = chess.square_rank(from_square)
        from_file = chess.square_file(from_square)

        # Directions for queen moves (horizontal, vertical, diagonal)
        queen_directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # Rook-like
            (1, 1), (-1, -1), (1, -1), (-1, 1)  # Bishop-like
        ]

        for dx, dy in queen_directions:
            x, y = from_file, from_rank
            while True:
                x += dx
                y += dy
                if 0 <= x < 8 and 0 <= y < 8:
                    to_square = chess.square(x, y)
                    move = chess.Move(from_square, to_square)
                    if move not in index_to_move.values():
                        index_to_move[index] = move
                        index += 1
                else:
                    break

        # Knight-like jumps
        knight_offsets = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]
        for dx, dy in knight_offsets:
            x = from_file + dx
            y = from_rank + dy
            if 0 <= x < 8 and 0 <= y < 8:
                to_square = chess.square(x, y)
                move = chess.Move(from_square, to_square)
                if move not in index_to_move.values():
                    index_to_move[index] = move
                    index += 1

    # Step 3: Pawn promotions from 7th → 8th rank and 2nd → 1st rank
    promotion_ranks = {
        chess.WHITE: (6, 7),  # rank 7 → 8 (index 6 → 7)
        chess.BLACK: (1, 0),  # rank 2 → 1 (index 1 → 0)
    }
    promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

    for color in [chess.WHITE, chess.BLACK]:
        from_rank, to_rank = promotion_ranks[color]
        for file in range(8):
            from_square = chess.square(file, from_rank)

            # Straight promotion (no capture)
            to_square = chess.square(file, to_rank)
            for promo in promotion_pieces:
                move = chess.Move(from_square, to_square, promotion=promo)
                if move not in index_to_move.values():
                    index_to_move[index] = move
                    index += 1

            # Diagonal capture promotions
            for df in [-1, 1]:  # Left and right diagonal
                capture_file = file + df
                if 0 <= capture_file < 8:
                    to_square = chess.square(capture_file, to_rank)
                    for promo in promotion_pieces:
                        move = chess.Move(from_square, to_square, promotion=promo)
                        if move not in index_to_move.values():
                            index_to_move[index] = move
                            index += 1

    # OPTIONAL print(f"Total moves: {index}")
    return index_to_move


def batch_board_to_tensor(boards: list) -> torch.Tensor:
    """
    Batch encode multiple chess boards into tensors.
    More efficient than calling board_to_tensor() individually.
    
    Args:
        boards: List of chess.Board objects
    
    Returns:
        Tensor of shape [len(boards), 855]
    """
    batch_size = len(boards)
    tensors = torch.zeros(batch_size, 855, dtype=torch.float32)
    
    # Piece type → index in one-hot (0–12)
    piece_to_index = {
        None: 0,
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6,
    }
    
    for batch_idx, board in enumerate(boards):
        # Fill piece encoding (64 × 13 = 832)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            base = square * 13
            
            if piece is None:
                tensors[batch_idx, base] = 1.0
            else:
                offset = piece_to_index[piece.piece_type]
                if piece.color == chess.BLACK:
                    offset += 6
                tensors[batch_idx, base + offset] = 1.0
        
        # Metadata
        tensors[batch_idx, 832] = float(board.turn)
        tensors[batch_idx, 833] = float(board.has_kingside_castling_rights(chess.WHITE))
        tensors[batch_idx, 834] = float(board.has_queenside_castling_rights(chess.WHITE))
        tensors[batch_idx, 835] = float(board.has_kingside_castling_rights(chess.BLACK))
        tensors[batch_idx, 836] = float(board.has_queenside_castling_rights(chess.BLACK))
        tensors[batch_idx, 837] = min(board.halfmove_clock, 100) / 100.0
        
        # En passant
        ep_square = board.ep_square
        if ep_square is not None:
            file = chess.square_file(ep_square)
            rank = chess.square_rank(ep_square)
            if rank == 5:
                index = file
            elif rank == 2:
                index = 8 + file
            else:
                index = None
            if index is not None:
                tensors[batch_idx, 838 + index] = 1.0
        
        # Repetition
        tensors[batch_idx, 854] = float(board.is_repetition(2))
    
    return tensors


index_to_move = build_move_index()
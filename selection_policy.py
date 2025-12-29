import torch
import torch.nn.functional as F
from typing import List
from tensor_game_conversion import index_to_move

# Build move → index dictionary
move_to_index = {m: i for i, m in index_to_move.items()}

def select_move(
    logits: torch.Tensor,
    legal_moves: List,
    move_num: int = 0,
    deterministic: bool = False,
    temperature: float = 1.0,
    early_move_random_top_k: int = 10,  # Increased from 5
    early_move_threshold: int = 12,  # Increased from 4
    value_pred: torch.Tensor = None,
    use_value_selection: bool = False,
    value_weight: float = 0.5
) -> int:
    """
    Move selection policy: only operates on legal moves.
    
    Args:
        logits: Policy logits for all moves
        legal_moves: List of legal chess.Move objects
        move_num: Current move number in game
        deterministic: If True, always pick best move
        temperature: Temperature for softmax sampling
        early_move_random_top_k: For early moves, pick randomly from top-K
        early_move_threshold: First N moves considered "early game"
        value_pred: Optional value prediction for current position (scalar)
        use_value_selection: If True, combine policy logits with value predictions
        value_weight: Weight for value component when use_value_selection=True (0-1)
    """
    # Convert legal_moves → indices
    legal_indices = [move_to_index[m] for m in legal_moves if m in move_to_index]
    if not legal_indices:
        raise RuntimeError("No legal moves mapped to indices!")

    # Extract logits for legal moves only
    legal_logits = logits[legal_indices]
    
    # If value-based selection is enabled, we need to estimate values for each legal move
    # For now, we'll use a simple approach: combine policy logits with value prediction
    # In the future, this could be enhanced to look ahead and evaluate each move
    if use_value_selection and value_pred is not None:
        # Combine policy logits with value prediction
        # Higher value = better position, so add value to logits (weighted)
        # Note: value_pred is from current position, not after each move
        # This is a simple heuristic - in future could evaluate each move position
        value_bonus = value_pred.item() * value_weight
        # Add value bonus to all legal moves (simple approach)
        # In a more sophisticated version, we'd evaluate each move's resulting position
        legal_logits = legal_logits + value_bonus

    # Early moves: stochastic top-K
    if move_num < early_move_threshold and not deterministic:
        k = min(early_move_random_top_k, len(legal_logits))
        if k == 1:
            move_index = legal_indices[int(torch.argmax(legal_logits).item())]
        else:
            topk_values, topk_indices = torch.topk(legal_logits, k=k)
            chosen = topk_indices[torch.randint(k, (1,)).item()]
            move_index = legal_indices[int(chosen.item())]
        return move_index

    # Mid-game exploration: moves 12-30 use slightly higher temperature
    mid_game_threshold = 30
    effective_temperature = temperature
    if move_num < mid_game_threshold and not deterministic:
        effective_temperature = temperature * 1.2  # 20% higher temperature for mid-game

    # Later moves: deterministic or softmax sampling
    if deterministic:
        move_index = legal_indices[int(torch.argmax(legal_logits).item())]
    else:
        # softmax over legal moves only
        probs = F.softmax(legal_logits / max(effective_temperature, 1e-6), dim=0)
        chosen = torch.multinomial(probs, num_samples=1).item()
        move_index = legal_indices[int(chosen)]

    return move_index

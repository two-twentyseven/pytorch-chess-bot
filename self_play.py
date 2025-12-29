import torch
import chess
from typing import List, Dict, Any
from chess_gym.envs.chess_env import ChessEnv
from chess_policy_network import ChessPolicyNetwork
from tensor_game_conversion import board_to_tensor, index_to_move
from selection_policy import select_move, move_to_index

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_AMP_SAFE = torch.cuda.is_available()


def calculate_draw_risk_penalty(board: chess.Board) -> float:
    """
    Calculate progressive draw penalty based on draw conditions.
    Returns negative penalty value (more negative = worse draw risk).
    """
    # Threefold repetition (position seen twice already)
    if board.is_repetition(2):
        return -0.8
    
    # Stalemate (no legal moves but not checkmate)
    if board.is_stalemate():
        return -0.7
    
    # Insufficient material to checkmate
    if board.is_insufficient_material():
        return -0.6
    
    # 75-move rule
    if board.is_seventyfive_moves():
        return -0.8
    
    # Approaching 50-move rule
    if board.halfmove_clock > 40:
        return -0.4
    
    # Position seen once (approaching threefold repetition)
    if board.is_repetition(1):
        return -0.3
    
    # Regular draw (other conditions)
    if board.is_game_over(claim_draw=True):
        return -0.5
    
    return 0.0  # No draw risk


def play_self_play_games_batch(
    policy_net: torch.nn.Module,
    num_games: int = 8,
    temperature: float = 1.0,
    max_moves: int = 1000,
    deterministic: bool = False,
    draw_penalty: float = -0.1,  # Kept for backward compatibility, but draw_risk_penalties are used instead
    early_move_threshold: int = 4,
    early_move_random_top_k: int = 5,
    use_value_selection: bool = False,
    value_weight: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Generate a batch of self-play trajectories for the training loop.
    Runs many ChessEnv instances in parallel on GPU-ready tensors to
    maximize throughput and returns per-game trajectories with rewards.
    """
    envs = [ChessEnv(observation_mode="piece_map") for _ in range(num_games)]
    active_mask = [True] * num_games
    move_counts = [0] * num_games
    trajectories: List[Dict[str, Any]] = [{
        "states": [], 
        "actions": [], 
        "log_probs": [],
        "values": [],  # NEW: store value predictions
        "legal_moves": [],  # NEW: store legal moves for each state
        "player_turns": [],  # NEW: track which player made each move (True=white, False=black)
        "draw_risk_penalties": []  # NEW: track draw risk penalties per move
    } for _ in range(num_games)]

    for env in envs:
        env.reset()
    states = [board_to_tensor(env.board).float().to(device) for env in envs]
    policy_net = policy_net.to(device).eval()

    while any(active_mask):
        active_indices = [i for i, active in enumerate(active_mask) if active and states[i] is not None]
        if not active_indices:
            break
        active_states = torch.stack([states[i] for i in active_indices])

        with torch.no_grad(), torch.amp.autocast(device_type=DEVICE_TYPE, enabled=USE_AMP_SAFE):
            logits_batched, values_batched = policy_net(active_states)
            values_batched = values_batched.squeeze(-1)  # [batch_size]

        for batch_idx, env_idx in enumerate(active_indices):
            env = envs[env_idx]
            board = env.board
            # Clone to avoid keeping reference to full batch tensor
            logits = logits_batched[batch_idx].clone()

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                active_mask[env_idx] = False
                continue

            # Get value prediction for current position (for value-based selection)
            current_value = values_batched[batch_idx] if use_value_selection else None
            
            move_index = select_move(
                logits=logits,
                legal_moves=legal_moves,
                move_num=move_counts[env_idx],
                deterministic=deterministic,
                temperature=temperature,
                early_move_random_top_k=early_move_random_top_k,
                early_move_threshold=early_move_threshold,
                value_pred=current_value,
                use_value_selection=use_value_selection,
                value_weight=value_weight
            )
            move = index_to_move[move_index]

            # Compute log_prob with gradients disabled, then detach immediately
            with torch.no_grad():
                probs = torch.softmax(logits / max(temperature, 1e-6), dim=0)
                log_prob = torch.log(probs[move_index] + 1e-8)
            
            # Store trajectory data: detach and move to CPU immediately to free GPU memory
            trajectories[env_idx]["states"].append(states[env_idx].detach().cpu())
            trajectories[env_idx]["actions"].append(int(move_index))
            trajectories[env_idx]["log_probs"].append(log_prob.detach().cpu())
            trajectories[env_idx]["values"].append(values_batched[batch_idx].detach().cpu())
            trajectories[env_idx]["legal_moves"].append(legal_moves)  # Store legal moves
            trajectories[env_idx]["player_turns"].append(board.turn)  # True=white, False=black

            _, reward, terminated, truncated, _ = env.step(move)
            move_counts[env_idx] += 1
            
            # Calculate draw risk penalty after the move
            draw_penalty = calculate_draw_risk_penalty(env.board)
            trajectories[env_idx]["draw_risk_penalties"].append(draw_penalty)

            if terminated or truncated or move_counts[env_idx] >= max_moves:
                active_mask[env_idx] = False
                # Clear reference to state tensor for finished games
                states[env_idx] = None
            else:
                # Refresh encoded board for next step
                states[env_idx] = board_to_tensor(env.board).float().to(device)

    results = []
    for env_idx, env in enumerate(envs):
        final_board = env.board
        
        # Determine final game outcome
        if final_board.is_checkmate():
            winner_is_white = not final_board.turn
            final_reward = 1.0 if winner_is_white else -1.0
        else:
            # Draw or stalemate
            final_reward = draw_penalty

        traj = trajectories[env_idx]
        
        # Assign rewards per move based on which player made the move
        # Incorporate draw risk penalties during the game
        traj["rewards"] = []
        for move_idx, (player_was_white, draw_risk) in enumerate(zip(traj["player_turns"], traj["draw_risk_penalties"])):
            # Base reward from game outcome
            if final_reward == 1.0:  # White won
                move_reward = 1.0 if player_was_white else -1.0
            elif final_reward == -1.0:  # Black won
                move_reward = 1.0 if not player_was_white else -1.0
            else:  # Draw
                move_reward = -0.5  # Base draw penalty
            
            # Add draw risk penalty (negative value, so it reduces reward)
            move_reward += draw_risk
            
            traj["rewards"].append(move_reward)
        
        # Keep backward compatibility: also store single reward for legacy code
        traj["reward"] = final_reward
        
        results.append(traj)

    return results

def play_single_game_pgn(
    policy_net: torch.nn.Module,
    temperature: float = 1.0,
    max_moves: int = 1000,
    deterministic: bool = True,
    stochastic_first_moves: int = 0,
    early_move_random_top_k: int = 5
) -> str:
    """
    Play one game with the current policy and return a PGN object.
    Useful for qualitative inspection after/between training runs.
    """
    from chess import pgn
    env = ChessEnv(observation_mode="piece_map")
    env.reset()
    board = env.board
    policy_net = policy_net.to(device).eval()

    move_history = []

    for move_num in range(max_moves):
        state = board_to_tensor(board).float().to(device).unsqueeze(0)

        with torch.no_grad(), torch.amp.autocast(device_type=DEVICE_TYPE, enabled=USE_AMP_SAFE):
            logits_batched, _ = policy_net(state)
        logits = logits_batched.squeeze(0)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break

        # For display games: first move sampled from softmax distribution over logits
        if move_num == 0 and stochastic_first_moves > 0:
            # Sample from softmax distribution for first move
            import torch.nn.functional as F
            legal_indices = [move_to_index[m] for m in legal_moves if m in move_to_index]
            if not legal_indices:
                raise RuntimeError("No legal moves mapped to indices!")
            legal_logits = logits[legal_indices]
            probs = F.softmax(legal_logits / max(temperature, 1e-6), dim=0)
            chosen = torch.multinomial(probs, num_samples=1).item()
            move_index = legal_indices[int(chosen)]
        else:
            # For display games: randomize first move(s) regardless of deterministic flag
            is_deterministic = deterministic and move_num >= stochastic_first_moves
            move_index = select_move(
                logits=logits,
                legal_moves=legal_moves,
                move_num=move_num,
                deterministic=is_deterministic,
                temperature=temperature,
                early_move_random_top_k=early_move_random_top_k,
                early_move_threshold=stochastic_first_moves
            )
        move = index_to_move[move_index]
        move_history.append(move)

        _, _, terminated, truncated, _ = env.step(move)
        if terminated or truncated:
            break
        board = env.board

    # Build PGN efficiently
    pgn_game = pgn.Game()
    pgn_game.setup(chess.Board())
    node = pgn_game
    for move in move_history:
        node = node.add_variation(move)

    return pgn_game

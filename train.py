import torch
import torch.nn.functional as F
from collections import deque
import random
from chess_policy_network import ChessPolicyNetwork, mask_illegal_moves, batch_mask_illegal_moves
from self_play import play_self_play_games_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_AMP_SAFE = torch.cuda.is_available()  # FP16 only if CUDA available


class ReplayBuffer:
    """Stores trajectories for policy gradient updates."""
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, trajectory):
        """trajectory is a dict with keys: states, actions, log_probs, reward
        States and log_probs should already be detached CPU tensors."""
        self.buffer.append(trajectory)

    def sample(self, batch_size):
        # Tensors are already detached CPU tensors from add()
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)


def train_iteration(policy_net, optimizer, replay_buffer, batch_size=16,
                    value_coeff=1.0, entropy_coeff=0.15, clip_grad_norm=1.0, scaler=None):
    """
    Train on a batch of trajectories using REINFORCE with value baseline and entropy regularization.
    
    Key improvements:
    - Per-move reward signals (not averaged)
    - Value baseline (advantage) to reduce variance
    - Entropy regularization to prevent policy collapse
    - Proper gradient accumulation across batch
    - Illegal move masking during training
    """
    if len(replay_buffer) < batch_size:
        return None

    batch = replay_buffer.sample(batch_size)
    policy_net.train()
    optimizer.zero_grad()

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy_loss = 0.0
    total_moves = 0

    for traj in batch:
        # Stack CPU tensors and move to device for training
        states = torch.stack(traj["states"]).to(device)
        actions = torch.tensor(traj["actions"], dtype=torch.long, device=device)
        
        # Use per-move rewards if available, otherwise fall back to single reward (backward compatibility)
        if "rewards" in traj and len(traj["rewards"]) > 0:
            rewards = torch.tensor(traj["rewards"], dtype=torch.float32, device=device)  # [num_moves]
        else:
            # Backward compatibility: use single reward for all moves
            reward = traj["reward"]
            rewards = torch.full((len(actions),), reward, dtype=torch.float32, device=device)
        
        # Get legal moves for each state (now stored in trajectory)
        legal_moves_list = traj.get("legal_moves", None)

        with torch.amp.autocast(device_type=DEVICE_TYPE, enabled=(scaler is not None and USE_AMP_SAFE)):
            logits, values_pred = policy_net(states)
            values_pred = values_pred.squeeze(-1)  # [num_moves]

            # Mask illegal moves if we have legal moves info (vectorized)
            if legal_moves_list is not None:
                logits = batch_mask_illegal_moves(logits, legal_moves_list)

            probs = F.softmax(logits, dim=-1)
            
            # Compute log probabilities for selected actions
            # Use advanced indexing: probs[batch_idx, action_idx]
            batch_indices = torch.arange(len(actions), device=device)
            selected_log_probs = torch.log(probs[batch_indices, actions] + 1e-8)
            
            # Compute advantage: rewards - values (detached to prevent value gradient from affecting policy)
            # rewards and values_pred are both [num_moves]
            advantage = rewards - values_pred.detach()
            
            # Policy loss: -log_prob * advantage (per move, then sum)
            # Positive advantage (good move) increases probability, negative decreases it
            policy_loss = -(selected_log_probs * advantage).sum()
            
            # Value loss: MSE between predicted and actual rewards (per move)
            value_loss = F.mse_loss(values_pred, rewards)
            
            # Entropy regularization: encourage exploration
            # Entropy = -sum(p * log(p)), we want to maximize it (so subtract in loss)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            entropy_loss = -entropy_coeff * entropy  # Negative because we want to maximize entropy
            
            # Total loss
            loss = policy_loss + value_coeff * value_loss + entropy_loss

        # Accumulate gradients (don't step optimizer yet)
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy_loss += entropy_loss.item()
        total_moves += len(actions)

    # Apply gradient clipping and optimizer step once for entire batch
    if clip_grad_norm:
        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), clip_grad_norm)
    
    if scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    return {
        "loss": (total_policy_loss + value_coeff * total_value_loss + total_entropy_loss) / batch_size,
        "policy_loss": total_policy_loss / batch_size,
        "value_loss": total_value_loss / batch_size,
        "entropy_loss": total_entropy_loss / batch_size,
        "avg_moves_per_traj": total_moves / batch_size
    }


def run_training_loop(policy_net, optimizer, iterations=100, games_per_iter=16,
                      batch_size=16, epochs_per_iter=1, value_coeff=1.0,
                      entropy_start=0.15, entropy_end=0.045,
                      save_path="policy_net.pt", use_amp=True,
                      temperature_start=1.5, temperature_end=1.0,
                      early_move_threshold=12, early_move_random_top_k=10):
    """
    Run self-play + training loop with modular exploration parameters.
    """
    replay_buffer = ReplayBuffer(capacity=50000)
    policy_net = policy_net.to(device)
    scaler = torch.amp.GradScaler(enabled=(use_amp and USE_AMP_SAFE))

    # --- Warm-up replay buffer to avoid empty sampling ---
    if len(replay_buffer) < batch_size:
        print(f"Warm-up: populating replay buffer with {batch_size} games...")
        warmup_games = play_self_play_games_batch(
            policy_net,
            num_games=batch_size,
            temperature=temperature_start,
            max_moves=400,
            deterministic=False,
            draw_penalty=-0.1,  # Legacy parameter, draw_risk_penalties are used instead
            early_move_threshold=early_move_threshold,
            early_move_random_top_k=early_move_random_top_k
        )
        for g in warmup_games:
            replay_buffer.add(g)
    # --- End warm-up ---

    for it in range(1, iterations + 1):
        # Linear scheduling for entropy and temperature
        alpha = (it - 1) / max(1, iterations - 1)
        entropy_coeff = entropy_start * (1 - alpha * 0.7) + entropy_end * (alpha * 0.7)
        temperature = temperature_start * (1 - alpha) + temperature_end * alpha

        # Self-play batch
        games = play_self_play_games_batch(
            policy_net,
            num_games=games_per_iter,
            temperature=temperature,
            max_moves=400,
            deterministic=False,
            draw_penalty=-0.1,  # Legacy parameter, draw_risk_penalties are used instead
            early_move_threshold=early_move_threshold,
            early_move_random_top_k=early_move_random_top_k
        )

        for g in games:
            replay_buffer.add(g)

        stats = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0}
        for _ in range(epochs_per_iter):
            res = train_iteration(policy_net, optimizer, replay_buffer,
                                  batch_size=batch_size, value_coeff=value_coeff,
                                  entropy_coeff=entropy_coeff,
                                  clip_grad_norm=1.0, scaler=scaler)
            if res is None:
                continue
            stats["loss"] += res["loss"]
            stats["policy_loss"] += res["policy_loss"]
            stats["value_loss"] += res["value_loss"]
            stats["entropy_loss"] += res.get("entropy_loss", 0.0)

        stats = {k: v/max(1, epochs_per_iter) for k,v in stats.items()}

        torch.save(policy_net.state_dict(), save_path)
        print(f"[iter {it}] loss={stats['loss']:.4f} "
              f"policy={stats['policy_loss']:.4f} value={stats['value_loss']:.4f} "
              f"entropy={stats['entropy_loss']:.4f} "
              f"entropy_coeff={entropy_coeff:.4f} temp={temperature:.3f} games={len(games)} saved={save_path}")

    return policy_net


if __name__ == "__main__":
    net = ChessPolicyNetwork().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-4)

    trained = run_training_loop(
        net,
        optimizer,
        iterations=3,
        games_per_iter=8,
        batch_size=8,
        epochs_per_iter=1,
        use_amp=True,
        entropy_start=0.15,
        entropy_end=0.045,
        temperature_start=1.5,
        temperature_end=1.0,
        early_move_threshold=12,
        early_move_random_top_k=10
    )
    print("Training loop finished")

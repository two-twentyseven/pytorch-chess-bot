import torch
import torch.optim as optim
from chess_policy_network import ChessPolicyNetwork
from train import run_training_loop
from self_play import play_single_game_pgn

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # 1) Network architecture parameter
    use_conv = True             # use convolutional layers for spatial pattern recognition
    
    # 2) Initialize network
    net = ChessPolicyNetwork(use_conv=use_conv).to(device)

    # 3) Optimizer
    optimizer = optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-4)

    # 4) Training parameters
    iterations = 50             # number of training iterations
    games_per_iter = 64         # self-play games per iteration
    batch_size = 32            # batch size for gradient updates
    epochs_per_iter = 2         # training passes per iteration
    value_coeff = 1.0           # weighting of value loss
    entropy_start = 0.15       # entropy regularization start (high exploration)
    entropy_end = 0.045        # entropy regularization end (decays over training)
    save_path = "policy_net_extensive.pt"
    temperature_start = 1.5    # temperature for self-play exploration (start high)
    temperature_end = 1.0       # temperature for self-play exploration (decays to 1.0)
    early_move_threshold = 12   # first N moves considered "early game" (increased for more exploration)
    early_move_random_top_k = 10  # pick randomly among top-K moves early game (increased for diversity)

    # 5) Run training loop
    trained_net = run_training_loop(
        policy_net=net,
        optimizer=optimizer,
        iterations=iterations,
        games_per_iter=games_per_iter,
        batch_size=batch_size,
        epochs_per_iter=epochs_per_iter,
        value_coeff=value_coeff,
        entropy_start=entropy_start,
        entropy_end=entropy_end,
        save_path=save_path,
        use_amp=True,
        temperature_start=temperature_start,
        temperature_end=temperature_end,
        early_move_threshold=early_move_threshold,
        early_move_random_top_k=early_move_random_top_k
    )
    
    # Export final trained model (explicit save after training completes)
    torch.save(trained_net.state_dict(), save_path)
    print(f"\n✓ Model exported successfully to: {save_path}")
    print(f"  Architecture: use_conv={use_conv}")
    print(f"  Model can be loaded with: ChessPolicyNetwork(use_conv={use_conv}).load_state_dict(torch.load('{save_path}'))\n")

    # 6) Test/game parameters
    num_test_games = 10               # number of PGN showcase games
    max_moves = 400                   # max moves per game
    deterministic = True              # deterministic moves after early stochastic phase
    stochastic_first_moves = 1        # number of stochastic early moves in demo (just first move)

    # 7) Generate PGNs using modular selection policy
    pgn_games = []
    for i in range(num_test_games):
        print(f"Generating PGN for test game {i+1}/{num_test_games}...")
        pgn = play_single_game_pgn(
            policy_net=trained_net,
            max_moves=max_moves,
            deterministic=deterministic,
            stochastic_first_moves=stochastic_first_moves,
            early_move_random_top_k=early_move_random_top_k
        )
        pgn_games.append(pgn)

    # 8) Print PGNs
    for i, pgn in enumerate(pgn_games):
        print(f"\n===== TEST GAME {i+1} PGN =====\n")
        print(pgn)
        print("\n===== END OF GAME =====\n")

    print("Extensive training + test finished")

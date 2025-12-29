import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from typing import List
from tensor_game_conversion import board_to_tensor, index_to_move

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reverse mapping for masking
move_to_index = {move: idx for idx, move in index_to_move.items()}


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """Proper residual block with LayerNorm and dropout."""
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return F.relu(x + residual)


class SelfAttentionBlock(nn.Module):
    """Multi-head self-attention for long-range dependencies."""
    def __init__(self, dim, num_heads=8, dropout=0.15):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: [batch, seq_len, dim] or [batch, dim] for single position
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, dim]
        
        residual = x
        x = self.norm1(x)
        
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        
        return x.squeeze(1) if x.shape[1] == 1 else x


class ChessPolicyNetwork(nn.Module):
    def __init__(self, dropout: float = 0.15, use_conv: bool = True):
        """
        Improved chess policy + value network with convolutional layers for spatial pattern recognition.
        
        Architecture:
        - Convolutional layers to capture local patterns (attacks, defenses, piece structures)
        - MLP layers for global strategy integration
        - Shared backbone with separate policy/value heads
        
        Args:
            dropout: Dropout probability
            use_conv: If True, use convolutional layers. If False, use original MLP-only architecture.
        """
        super().__init__()
        self.use_conv = use_conv
        self.dropout = nn.Dropout(p=dropout)
        
        if use_conv:
            # Reshape: 832 dims (64 squares × 13 channels) → 8×8×13
            # Metadata: 23 dims (turn, castling, halfmove, en_passant, repetition)
            
            # Multi-scale convolutions for different pattern sizes
            # 3x3 convs for local patterns
            self.conv1_3x3 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
            self.norm_conv1_3x3 = nn.BatchNorm2d(64)
            
            # 5x5 convs for broader patterns
            self.conv1_5x5 = nn.Conv2d(13, 64, kernel_size=5, padding=2)
            self.norm_conv1_5x5 = nn.BatchNorm2d(64)
            
            # Concatenate multi-scale features: 64 + 64 = 128 channels
            self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 8×8×256
            self.norm_conv2 = nn.BatchNorm2d(256)
            self.se1 = SqueezeExcitation(256, reduction=16)
            
            self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # 8×8×512
            self.norm_conv3 = nn.BatchNorm2d(512)
            self.se2 = SqueezeExcitation(512, reduction=16)
            
            # Flatten conv output: 8×8×512 = 32768
            # Project down to manageable size before attention (reduces memory)
            self.attention_proj_down = nn.Linear(32768, 1024)
            
            # Use self-attention on reduced features (much more memory efficient)
            self.attention = SelfAttentionBlock(1024, num_heads=8, dropout=dropout)
            
            # Project back up for concatenation with metadata
            self.attention_proj_up = nn.Linear(1024, 8192)
            
            # Plus metadata: 23
            mlp_input_size = 8192 + 23
        else:
            # Original MLP architecture
            mlp_input_size = 855

        # MLP layers with proper residual blocks
        self.fc1 = nn.Linear(mlp_input_size, 1024)
        self.residual1 = ResidualBlock(1024, dropout=dropout)
        self.residual2 = ResidualBlock(1024, dropout=dropout)
        self.residual3 = ResidualBlock(1024, dropout=dropout)
        
        # Final projection to 512 for heads
        self.fc_final = nn.Linear(1024, 512)

        # Output heads
        self.policy_head = nn.Linear(512, len(index_to_move))
        self.value_head = nn.Linear(512, 1)

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        Args:
            x: [batch, 855] tensor (flat board representation)
        Returns:
            policy_logits: [batch, num_moves]
            value: [batch, 1], in [-1, 1]
        """
        # Extract turn indicator early (before any processing)
        # Turn is always at position 832 in the input tensor (1.0 = white, 0.0 = black)
        turn = x[:, 832:833]  # [batch, 1]
        
        if self.use_conv:
            # Split input: piece encoding (832) + metadata (23)
            batch_size = x.shape[0]
            piece_encoding = x[:, :832]  # [batch, 832]
            metadata = x[:, 832:]  # [batch, 23]
            
            # Reshape piece encoding to 8×8×13: [batch, 832] → [batch, 8, 8, 13]
            # The encoding is: square 0 (0-12), square 1 (13-25), ..., square 63 (819-831)
            # Use reshape instead of view to handle non-contiguous tensors
            board_2d = piece_encoding.reshape(batch_size, 8, 8, 13)
            # Convert to NCHW format: [batch, 13, 8, 8]
            board_2d = board_2d.permute(0, 3, 1, 2)
            
            # Multi-scale convolutions
            x_3x3 = F.relu(self.norm_conv1_3x3(self.conv1_3x3(board_2d)))
            x_5x5 = F.relu(self.norm_conv1_5x5(self.conv1_5x5(board_2d)))
            x_conv = torch.cat([x_3x3, x_5x5], dim=1)  # [batch, 128, 8, 8]
            x_conv = self.dropout(x_conv)
            
            # Second conv layer with SE
            x_conv = F.relu(self.norm_conv2(self.conv2(x_conv)))  # [batch, 256, 8, 8]
            x_conv = self.se1(x_conv)
            x_conv = self.dropout(x_conv)
            
            # Third conv layer with SE
            x_conv = F.relu(self.norm_conv3(self.conv3(x_conv)))  # [batch, 512, 8, 8]
            x_conv = self.se2(x_conv)
            x_conv = self.dropout(x_conv)
            
            # Flatten conv output: [batch, 512, 8, 8] → [batch, 32768]
            x_conv_flat = x_conv.reshape(batch_size, -1)  # [batch, 32768]
            
            # Project down to manageable size before attention
            x_proj_down = self.attention_proj_down(x_conv_flat)  # [batch, 1024]
            x_proj_down = F.relu(x_proj_down)
            
            # Self-attention for long-range dependencies (on reduced dimension)
            x_attn = self.attention(x_proj_down)  # [batch, 1024]
            
            # Project back up for concatenation with metadata
            x_attn = self.attention_proj_up(x_attn)  # [batch, 8192]
            x_attn = F.relu(x_attn)
            
            # Concatenate with metadata
            x = torch.cat([x_attn, metadata], dim=1)  # [batch, 8192 + 23]
        else:
            # Original MLP path
            x = x
        
        # MLP layers with proper residual blocks
        x = F.relu(self.fc1(x))  # [batch, 1024]
        x = self.dropout(x)
        
        # Residual blocks
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        
        # Final projection
        x = self.fc_final(x)  # [batch, 512]
        x = self.dropout(x)

        policy_logits = self.policy_head(x)
        value_raw = self.value_head(x)
        
        # Fix value head perspective: use turn extracted at the start of forward()
        # Convert to perspective: 1.0 for white, -1.0 for black  
        perspective = 2.0 * turn - 1.0  # [batch, 1]
        
        # Calibrate value: scale by 2.0 for better sensitivity, add small bias to discourage draws
        value_calibrated = torch.tanh(value_raw * 2.0) - 0.1  # Bias pushes draws toward negative
        
        # Multiply by perspective so value is from current player's viewpoint
        value = value_calibrated * perspective
        
        return policy_logits, value


def mask_illegal_moves(policy_logits: torch.Tensor, legal_moves):
    """
    Vectorized masking of illegal moves.
    Args:
        policy_logits: [num_moves] or [batch, num_moves]
        legal_moves: list of chess.Move objects
    Returns:
        masked logits tensor of same shape
    """
    masked = torch.full_like(policy_logits, float('-inf'))
    for move in legal_moves:
        idx = move_to_index.get(move)
        if idx is not None:
            masked[..., idx] = policy_logits[..., idx]
    return masked


def batch_mask_illegal_moves(policy_logits: torch.Tensor, legal_moves_list: List[List]):
    """
    Batch mask illegal moves for multiple states.
    More efficient than calling mask_illegal_moves() in a loop.
    
    Args:
        policy_logits: [batch_size, num_moves] tensor
        legal_moves_list: List of lists, each containing legal moves for one state
    Returns:
        masked logits tensor of same shape [batch_size, num_moves]
    """
    batch_size = policy_logits.shape[0]
    num_moves = policy_logits.shape[1]
    masked = torch.full_like(policy_logits, float('-inf'))
    
    # Pre-compute all legal move indices
    for batch_idx, legal_moves in enumerate(legal_moves_list):
        for move in legal_moves:
            idx = move_to_index.get(move)
            if idx is not None:
                masked[batch_idx, idx] = policy_logits[batch_idx, idx]
    
    return masked

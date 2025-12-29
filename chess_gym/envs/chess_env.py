import gymnasium as gym
from gymnasium import spaces

import chess
import chess.svg

import numpy as np

from io import BytesIO
import cairosvg
from PIL import Image


class MoveSpace(spaces.Space):
    def __init__(self, board):
        super().__init__(shape=(), dtype=object)  # scalar space, dtype=object for chess.Move
        self.board = board

    def sample(self):
        return np.random.choice(list(self.board.legal_moves))

    def contains(self, x):
        # Check if x is a legal move
        return isinstance(x, chess.Move) and x in self.board.legal_moves

    def __repr__(self):
        return "MoveSpace(legal moves of current board)"

    def seed(self, seed=None):
        np.random.seed(seed)


class ChessEnv(gym.Env):
    """Chess Environment"""
    metadata = {
        'render_modes': ['rgb_array', 'human'],
        'observation_modes': ['rgb_array', 'piece_map'],
        'render_fps': 30
    }

    def __init__(self, render_size=512, observation_mode='rgb_array', claim_draw=True, **kwargs):
        super(ChessEnv, self).__init__()

        if observation_mode == 'rgb_array':
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(render_size, render_size, 3),
                dtype=np.uint8)
        elif observation_mode == 'piece_map':
            self.observation_space = spaces.Box(
                low=-6, high=6,
                shape=(8, 8),
                dtype=np.int8)  # MUST be int8 to match returned array dtype
        else:
            raise Exception("observation_mode must be either 'rgb_array' or 'piece_map'")

        self.observation_mode = observation_mode

        self.chess960 = kwargs.get('chess960', False)
        self.board = chess.Board(chess960=self.chess960)

        if self.chess960:
            self.board.set_chess960_pos(np.random.randint(0, 960))

        self.render_size = render_size
        self.claim_draw = claim_draw

        self.viewer = None

        self.action_space = MoveSpace(self.board)

    def _get_image(self):
        out = BytesIO()
        bytestring = chess.svg.board(self.board, size=self.render_size).encode('utf-8')
        cairosvg.svg2png(bytestring=bytestring, write_to=out)
        image = Image.open(out).convert('RGB')  # ensure 3 channels, uint8
        arr = np.asarray(image, dtype=np.uint8)
        # Confirm shape matches observation space
        if arr.shape != self.observation_space.shape:
            raise ValueError(f"Image observation shape {arr.shape} does not match observation_space {self.observation_space.shape}")
        if arr.dtype != self.observation_space.dtype:
            raise ValueError(f"Image observation dtype {arr.dtype} does not match observation_space {self.observation_space.dtype}")
        return arr

    def _get_piece_configuration(self):
        piece_map = np.zeros(64, dtype=np.int8)  # must match observation_space dtype exactly
        for square, piece in self.board.piece_map().items():
            piece_map[square] = piece.piece_type * (piece.color * 2 - 1)
        arr = piece_map.reshape((8, 8))
        if arr.shape != self.observation_space.shape:
            raise ValueError(f"Piece configuration shape {arr.shape} does not match observation_space {self.observation_space.shape}")
        if arr.dtype != self.observation_space.dtype:
            raise ValueError(f"Piece configuration dtype {arr.dtype} does not match observation_space {self.observation_space.dtype}")
        return arr

    def _observe(self):
        if self.observation_mode == 'rgb_array':
            return self._get_image()
        else:
            return self._get_piece_configuration()

    def step(self, action):
        self.board.push(action)

        observation = self._observe()
        result = self.board.result()
        reward = 1 if result == '1-0' else -1 if result == '0-1' else 0
        terminated = self.board.is_game_over(claim_draw=self.claim_draw)
        truncated = False

        info = {
            'turn': self.board.turn,
            'castling_rights': self.board.castling_rights,
            'fullmove_number': self.board.fullmove_number,
            'halfmove_clock': self.board.halfmove_clock,
            'promoted': getattr(self.board, 'promoted', None),
            'chess960': self.board.chess960,
            'ep_square': self.board.ep_square
        }

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.board.reset()

        if self.chess960:
            self.board.set_chess960_pos(np.random.randint(0, 960))

        observation = self._observe()
        return observation, {}

    def render(self, mode='human'):
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from PIL import Image
            Image.fromarray(img).show()
            return True

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

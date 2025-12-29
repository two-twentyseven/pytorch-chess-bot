#!/usr/bin/env python3
"""
Interactive chess game against a trained model with PyQt6 GUI.
Usage: python user_play.py <model_path> [--use-conv True/False]

Installation:
    pip install PyQt6
    Or if using a virtual environment:
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate
    pip install PyQt6
"""

import sys
import os
import torch
import chess
from typing import Optional, List, Tuple

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QMessageBox, QDialog,
        QTextEdit
    )
    from PyQt6.QtCore import Qt, QTimer, QPoint, pyqtSignal
    from PyQt6.QtGui import QPainter, QColor, QFont, QMouseEvent, QPixmap
except ImportError:
    print("PyQt6 not found. Please install it with: pip install PyQt6")
    sys.exit(1)

from chess_policy_network import ChessPolicyNetwork
from tensor_game_conversion import board_to_tensor, index_to_move
from selection_policy import select_move

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str, use_conv: bool = True):
    """
    Load a trained model from a .pt file.
    
    Args:
        model_path: Path to the model file
        use_conv: Whether the model uses convolutional layers (must match training config)
    
    Returns:
        Loaded ChessPolicyNetwork model in eval mode
    """
    print(f"Loading model from: {model_path}")
    model = ChessPolicyNetwork(use_conv=use_conv).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✓ Model loaded successfully (use_conv={use_conv})")
    return model


def get_model_move(model: ChessPolicyNetwork, board: chess.Board, move_num: int) -> chess.Move:
    """
    Get the model's move for the current position.
    
    Args:
        model: The loaded policy network
        board: Current chess board position
        move_num: Current move number in the game
    
    Returns:
        chess.Move selected by the model
    """
    # Convert board to tensor
    state = board_to_tensor(board).float().to(device).unsqueeze(0)
    
    # Get model predictions
    with torch.no_grad():
        logits, _ = model(state)
    
    logits = logits.squeeze(0)
    legal_moves = list(board.legal_moves)
    
    # Select move using deterministic selection (best move)
    move_index = select_move(
        logits=logits,
        legal_moves=legal_moves,
        move_num=move_num,
        deterministic=True,
        temperature=1.0,
        early_move_random_top_k=1,
        early_move_threshold=0
    )
    
    move = index_to_move[move_index]
    return move


class ColorSelectionDialog(QDialog):
    """Dialog for selecting which color to play."""
    
    def __init__(self):
        super().__init__()
        self.user_plays_white = True
        self.setModal(True)  # Make dialog modal so it blocks and closes properly
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Select Your Color")
        self.setMinimumWidth(420)
        self.setMinimumHeight(220)
        layout = QVBoxLayout()
        layout.setSpacing(28)
        layout.setContentsMargins(32, 32, 32, 32)
        
        label = QLabel("Choose Your Side")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(
            "font-size: 20px; "
            "font-weight: 600; "
            "color: #1a1a1a; "
            "padding: 16px 8px; "
            "letter-spacing: 0.5px;"
        )
        layout.addWidget(label)
        
        subtitle = QLabel("Select which color you want to play:")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(
            "font-size: 14px; "
            "font-weight: 400; "
            "color: #6c757d; "
            "padding: 4px 8px 12px 8px;"
        )
        layout.addWidget(subtitle)
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(16)
        
        white_button = QPushButton("Play as White")
        white_button.clicked.connect(lambda: self.accept_with_color(True))
        white_button.setMinimumHeight(48)
        white_button.setMinimumWidth(140)
        button_layout.addWidget(white_button)
        
        black_button = QPushButton("Play as Black")
        black_button.clicked.connect(lambda: self.accept_with_color(False))
        black_button.setMinimumHeight(48)
        black_button.setMinimumWidth(140)
        button_layout.addWidget(black_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def accept_with_color(self, plays_white: bool):
        self.user_plays_white = plays_white
        self.accept()  # This closes the dialog and returns QDialog.DialogCode.Accepted


class ChessBoardWidget(QWidget):
    """Custom widget for displaying and interacting with the chess board."""
    
    move_made = pyqtSignal(chess.Move)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.board = chess.Board()
        # Square size will be computed dynamically on resize; start with a sensible default
        self.square_size = 80
        # Padding will scale with square size (calculated in resizeEvent)
        self.board_padding = 35  # Initial value, will be recalculated
        # Set size policy to allow expansion
        from PyQt6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 400)  # Larger minimum for better board visibility
        
        # Drag and drop state
        self.selected_square: Optional[int] = None
        self.drag_pos: Optional[QPoint] = None
        self.legal_moves: List[chess.Move] = []
        self.last_move: Optional[Tuple[int, int]] = None  # (from_square, to_square)
        self.flipped = False
        
        # Colors (reverted to original modern green/cream palette)
        self.light_square = QColor("#ebecd0")
        self.dark_square = QColor("#779952")
        self.legal_dot = QColor(63, 169, 245, 170)
        self.last_move_color = QColor(242, 232, 139, 180)
        self.selected_color = QColor(63, 169, 245, 90)

        # Piece images
        self.piece_pixmaps: dict[str, QPixmap] = {}
        self.scaled_pixmaps: dict[Tuple[str, int], QPixmap] = {}
        self.load_piece_pixmaps()
    
    def set_board(self, board: chess.Board):
        """Update the board state."""
        self.board = board
        self.update()
    
    def set_last_move(self, from_square: Optional[int], to_square: Optional[int]):
        """Set the last move for highlighting."""
        if from_square is not None and to_square is not None:
            self.last_move = (from_square, to_square)
        else:
            self.last_move = None
        self.update()
    
    def flip_board(self):
        """Flip the board view."""
        self.flipped = not self.flipped
        self.update()
    
    def square_at_position(self, pos: QPoint) -> Optional[int]:
        """Convert mouse position to chess square index."""
        x = pos.x() - self.board_padding
        y = pos.y() - self.board_padding
        
        if x < 0 or y < 0 or x >= 8 * self.square_size or y >= 8 * self.square_size:
            return None
        
        # Calculate display file and rank (screen coordinates)
        display_file = int(x / self.square_size)
        display_rank = int(y / self.square_size)
        
        # Convert display coordinates to chess board coordinates
        if self.flipped:
            file = 7 - display_file
            rank = 7 - display_rank
        else:
            file = display_file
            rank = 7 - display_rank  # Screen y=0 is top, chess rank 0 is bottom
        
        if 0 <= file < 8 and 0 <= rank < 8:
            return chess.square(file, rank)
        return None
    
    def square_to_rect(self, square: int) -> Tuple[int, int, int, int]:
        """Convert square index to screen rectangle."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        # Convert chess coordinates to display coordinates
        if self.flipped:
            display_file = 7 - file
            display_rank = 7 - rank
        else:
            display_file = file
            display_rank = 7 - rank  # Screen y=0 is top, chess rank 0 is bottom
        
        x = self.board_padding + display_file * self.square_size
        y = self.board_padding + display_rank * self.square_size
        
        return (x, y, self.square_size, self.square_size)
    
    def resizeEvent(self, event):
        """Recompute square size on resize so pieces fit cleanly."""
        # Calculate square size based on available space
        # Reserve space for coordinates (proportional padding)
        min_dimension = min(self.width(), self.height())
        # Estimate padding needed (will be refined after square_size calculation)
        estimated_padding = max(35, int(min_dimension * 0.08))
        
        available_width = max(1, self.width() - 2 * estimated_padding)
        available_height = max(1, self.height() - 2 * estimated_padding - 25)
        self.square_size = int(min(available_width, available_height) / 8)
        
        # Ensure minimum square size for visibility
        self.square_size = max(40, self.square_size)
        
        # Scale padding proportionally with square size, but with better minimum
        self.board_padding = max(30, int(self.square_size * 0.35))
        
        self.scaled_pixmaps.clear()
        super().resizeEvent(event)
        self.update()

    def load_piece_pixmaps(self):
        """Load piece assets from the pieces directory."""
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pieces")
        names = {
            "P": "wp", "p": "bp",
            "R": "wr", "r": "br",
            "N": "wn", "n": "bn",
            "B": "wb", "b": "bb",
            "Q": "wq", "q": "bq",
            "K": "wk", "k": "bk",
        }
        for key, fname in names.items():
            path_svg = os.path.join(base_dir, f"{fname}.svg")
            path_png = os.path.join(base_dir, f"{fname}.png")
            if os.path.exists(path_svg):
                self.piece_pixmaps[key] = QPixmap(path_svg)
            elif os.path.exists(path_png):
                self.piece_pixmaps[key] = QPixmap(path_png)

    def get_scaled_pixmap(self, key: str) -> Optional[QPixmap]:
        """Return a cached, scaled pixmap for the current square size."""
        if key not in self.piece_pixmaps:
            return None
        cache_key = (key, self.square_size)
        if cache_key in self.scaled_pixmaps:
            return self.scaled_pixmaps[cache_key]
        pix = self.piece_pixmaps[key].scaled(
            self.square_size, self.square_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.scaled_pixmaps[cache_key] = pix
        return pix

    def paintEvent(self, event):
        """Draw the chess board and pieces."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw coordinates
        font = QFont("Inter", max(11, int(self.square_size * 0.28)))
        painter.setFont(font)
        
        # Draw board squares
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, rank)
                
                if self.flipped:
                    display_file = 7 - file
                    display_rank = 7 - rank
                else:
                    display_file = file
                    display_rank = rank
                
                x = self.board_padding + display_file * self.square_size
                y = self.board_padding + (7 - display_rank) * self.square_size  # Screen y=0 is top, chess rank 0 is bottom
                
                # Determine square color (a1 should be dark, so file+rank even = dark)
                is_light = (file + rank) % 2 == 1
                color = self.light_square if is_light else self.dark_square
                painter.fillRect(x, y, self.square_size, self.square_size, color)
                
                # Last move overlay
                if self.last_move and square in self.last_move:
                    painter.fillRect(x, y, self.square_size, self.square_size, self.last_move_color)
                
                # Selected square overlay
                if self.selected_square == square:
                    painter.fillRect(x, y, self.square_size, self.square_size, self.selected_color)
                
                # Draw piece (hide from origin while dragging for cleaner feedback)
                piece = self.board.piece_at(square)
                if piece and not (self.selected_square == square and self.drag_pos is not None):
                    pix = self.get_scaled_pixmap(piece.symbol())
                    if pix:
                        px = x + (self.square_size - pix.width()) // 2
                        py = y + (self.square_size - pix.height()) // 2
                        painter.drawPixmap(px, py, pix)
                
                # Legal move dots
                if self.selected_square is not None:
                    for move in self.legal_moves:
                        if move.from_square == self.selected_square and move.to_square == square:
                            center_x = x + self.square_size // 2
                            center_y = y + self.square_size // 2
                            radius = max(6, int(self.square_size * 0.14))
                            painter.setBrush(self.legal_dot)
                            painter.setPen(Qt.PenStyle.NoPen)
                            painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        # Draw coordinates with better styling
        painter.setPen(QColor(108, 117, 125))  # Modern gray color #6c757d
        font = QFont("Inter", max(11, int(self.square_size * 0.18)))
        font.setWeight(500)  # Medium weight for better visibility
        painter.setFont(font)
        for i in range(8):
            file_label = chr(ord('a') + (7 - i if self.flipped else i))
            rank_label = str((i + 1) if self.flipped else (8 - i))
            
            # File labels (bottom) - centered below each square with proper spacing
            file_x = self.board_padding + i * self.square_size
            file_y = self.board_padding + 8 * self.square_size + 5
            painter.drawText(file_x, file_y, self.square_size, 20, 
                           Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop, file_label)
            
            # Rank labels (left) - centered to the left of each square with proper spacing
            rank_x = 5
            rank_y = self.board_padding + i * self.square_size
            painter.drawText(rank_x, rank_y, self.board_padding - 10, self.square_size,
                           Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignRight, rank_label)
        
        # Draw dragged piece if dragging
        if self.selected_square is not None and self.drag_pos is not None:
            piece = self.board.piece_at(self.selected_square)
            if piece:
                pix = self.get_scaled_pixmap(piece.symbol())
                if pix:
                    painter.drawPixmap(self.drag_pos.x() - pix.width() // 2,
                                       self.drag_pos.y() - pix.height() // 2,
                                       pix)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for piece selection."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        
        square = self.square_at_position(event.pos())
        if square is None:
            self.selected_square = None
            self.legal_moves = []
            self.update()
            return
        
        piece = self.board.piece_at(square)
        
        # Only allow selecting user's pieces (will be checked by parent)
        if piece:
            self.selected_square = square
            self.drag_pos = event.pos()
            
            # Get legal moves for selected piece
            self.legal_moves = [move for move in self.board.legal_moves if move.from_square == square]
            
            self.update()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for drag feedback."""
        if self.selected_square is not None:
            self.drag_pos = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release for move execution."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        
        if self.selected_square is None:
            return
        
        to_square = self.square_at_position(event.pos())
        
        if to_square is not None and to_square != self.selected_square:
            # Try to find a matching legal move
            move = None
            for m in self.legal_moves:
                if m.from_square == self.selected_square and m.to_square == to_square:
                    move = m
                    break
            
            if move:
                self.move_made.emit(move)
        
        # Reset selection
        self.selected_square = None
        self.drag_pos = None
        self.legal_moves = []
        self.update()


class ChessGameWindow(QMainWindow):
    """Main window for the chess game."""
    
    def __init__(self, model: ChessPolicyNetwork, user_plays_white: bool = True):
        super().__init__()
        self.model = model
        self.user_plays_white = user_plays_white
        self.board = chess.Board()
        self.move_count = 0
        self.move_history: List[Tuple[chess.Move, str, bool]] = []  # (move, SAN, is_user)
        self.board_history: List[chess.Board] = [chess.Board()]  # For undo functionality
        
        self.init_ui()
        # Color selection is handled in main() before window is shown
        self.update_display()
    
    def init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle("Chess AI - Play Against Machine Learning")
        self.setMinimumSize(1000, 750)  # Larger minimum for better board visibility
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(16)
        
        # Chess board - allow it to expand
        self.board_widget = ChessBoardWidget()
        self.board_widget.set_board(self.board)
        self.board_widget.move_made.connect(self.handle_user_move)
        main_layout.addWidget(self.board_widget, stretch=3, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Side panel - flexible but with max width, modern styling
        side_panel = QWidget()
        side_panel.setMaximumWidth(360)
        side_panel.setMinimumWidth(280)
        side_panel.setStyleSheet(
            "background-color: #ffffff; "
            "border-left: 1px solid #e9ecef;"
        )
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(24, 24, 24, 24)
        side_layout.setSpacing(20)
        main_layout.addWidget(side_panel, stretch=1)
        
        # Game status - modern card design
        status_header = QLabel("Game Status")
        status_header.setStyleSheet(
            "font-weight: 500; "
            "font-size: 13px; "
            "color: #6c757d; "
            "padding: 0px 0px 8px 0px; "
            "text-transform: uppercase; "
            "letter-spacing: 0.5px;"
        )
        side_layout.addWidget(status_header)
        
        self.status_label = QLabel("Game Status")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "background-color: #f8f9fa; "
            "border: none; "
            "border-radius: 8px; "
            "padding: 16px; "
            "font-weight: 500; "
            "font-size: 14px; "
            "color: #212529; "
            "min-height: 50px;"
        )
        side_layout.addWidget(self.status_label)
        side_layout.addSpacing(12)
        
        # FEN representation - modern, dynamic styling
        history_label = QLabel("Current Position")
        history_label.setStyleSheet(
            "font-weight: 500; "
            "font-size: 13px; "
            "color: #6c757d; "
            "padding: 12px 0px 6px 0px; "
            "text-transform: uppercase; "
            "letter-spacing: 0.5px;"
        )
        side_layout.addWidget(history_label)
        
        self.move_list = QTextEdit()
        self.move_list.setReadOnly(True)
        self.move_list.setMaximumHeight(100)
        self.move_list.setStyleSheet(
            "background-color: transparent; "
            "border: none; "
            "border-bottom: 1px solid #e9ecef; "
            "border-radius: 0px; "
            "padding: 8px 0px; "
            "font-family: 'SF Mono', 'Monaco', 'Courier New', monospace; "
            "font-size: 11px; "
            "color: #495057; "
            "line-height: 1.6;"
        )
        side_layout.addWidget(self.move_list)
        
        # Add spacing
        side_layout.addSpacing(8)
        
        # PGN notation - modern, dynamic styling
        notation_label = QLabel("Game Record")
        notation_label.setStyleSheet(
            "font-weight: 500; "
            "font-size: 13px; "
            "color: #6c757d; "
            "padding: 12px 0px 6px 0px; "
            "text-transform: uppercase; "
            "letter-spacing: 0.5px;"
        )
        side_layout.addWidget(notation_label)
        
        self.notation_text = QTextEdit()
        self.notation_text.setReadOnly(True)
        self.notation_text.setMaximumHeight(200)
        # Modern, clean styling without heavy borders
        self.notation_text.setStyleSheet(
            "background-color: transparent; "
            "border: none; "
            "border-bottom: 1px solid #e9ecef; "
            "border-radius: 0px; "
            "padding: 8px 0px; "
            "font-family: 'SF Mono', 'Monaco', 'Courier New', monospace; "
            "font-size: 11px; "
            "color: #495057; "
            "line-height: 1.6;"
        )
        side_layout.addWidget(self.notation_text)
        
        # Control buttons
        button_layout = QVBoxLayout()
        button_layout.setSpacing(12)
        
        self.undo_button = QPushButton("Undo Move")
        self.undo_button.clicked.connect(self.undo_move)
        button_layout.addWidget(self.undo_button)
        
        self.reset_button = QPushButton("Reset Game")
        self.reset_button.clicked.connect(self.reset_game)
        button_layout.addWidget(self.reset_button)
        
        self.flip_button = QPushButton("Flip Board")
        self.flip_button.clicked.connect(self.flip_board)
        button_layout.addWidget(self.flip_button)
        
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)
        button_layout.addWidget(self.quit_button)
        
        side_layout.addLayout(button_layout)
        side_layout.addStretch()
        
        main_layout.addWidget(side_panel)
        
        # Menu bar
        self.create_menu_bar()

        # Center window on screen after it has a size
        QTimer.singleShot(0, self.center_window)
    
    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        new_game_action = file_menu.addAction("New Game")
        new_game_action.triggered.connect(self.reset_game)
        file_menu.addSeparator()
        quit_action = file_menu.addAction("Quit")
        quit_action.triggered.connect(self.close)
        
        # View menu
        view_menu = menubar.addMenu("View")
        flip_action = view_menu.addAction("Flip Board")
        flip_action.triggered.connect(self.flip_board)

    def center_window(self):
        """Center the window on the primary screen."""
        screen = QApplication.primaryScreen()
        if not screen:
            return
        screen_geometry = screen.availableGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)
    
    def flip_board(self):
        """Flip the board view."""
        self.board_widget.flip_board()
    
    def update_display(self):
        """Update all display elements."""
        self.board_widget.set_board(self.board)
        self.update_status()
        self.update_move_history()
        self.update_notation()
        self.update_buttons()
    
    def update_status(self):
        """Update the game status display."""
        base_style = (
            "background-color: #ffffff; "
            "border: 1px solid #e9ecef; "
            "border-radius: 12px; "
            "padding: 16px; "
            "font-weight: 500; "
            "font-size: 14px; "
            "color: #1a1a1a;"
        )
        
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn else "White"
            status = f"CHECKMATE!\n{winner} wins!"
            self.status_label.setStyleSheet(
                "background-color: #fff5f5; "
                "border: none; "
                "border-left: 3px solid #dc3545; "
                "border-radius: 8px; "
                "padding: 16px; "
                "font-weight: 600; "
                "font-size: 14px; "
                "color: #c92a2a;"
            )
        elif self.board.is_stalemate():
            status = "STALEMATE!\nGame is a draw."
            self.status_label.setStyleSheet(
                "background-color: #fffbf0; "
                "border: none; "
                "border-left: 3px solid #ffc107; "
                "border-radius: 8px; "
                "padding: 16px; "
                "font-weight: 600; "
                "font-size: 14px; "
                "color: #e67700;"
            )
        elif self.board.is_insufficient_material():
            status = "DRAW\nInsufficient material"
            self.status_label.setStyleSheet(
                "background-color: #fffbf0; "
                "border: none; "
                "border-left: 3px solid #ffc107; "
                "border-radius: 8px; "
                "padding: 16px; "
                "font-weight: 600; "
                "font-size: 14px; "
                "color: #e67700;"
            )
        elif self.board.is_seventyfive_moves():
            status = "DRAW\n75-move rule"
            self.status_label.setStyleSheet(
                "background-color: #fffbf0; "
                "border: none; "
                "border-left: 3px solid #ffc107; "
                "border-radius: 8px; "
                "padding: 16px; "
                "font-weight: 600; "
                "font-size: 14px; "
                "color: #e67700;"
            )
        elif self.board.is_fivefold_repetition():
            status = "DRAW\nFivefold repetition"
            self.status_label.setStyleSheet(
                "background-color: #fffbf0; "
                "border: none; "
                "border-left: 3px solid #ffc107; "
                "border-radius: 8px; "
                "padding: 16px; "
                "font-weight: 600; "
                "font-size: 14px; "
                "color: #e67700;"
            )
        elif self.board.is_check():
            turn = "White" if self.board.turn else "Black"
            status = f"CHECK!\n{turn} is in check"
            self.status_label.setStyleSheet(
                "background-color: #e7f5ff; "
                "border: none; "
                "border-left: 3px solid #0d6efd; "
                "border-radius: 8px; "
                "padding: 16px; "
                "font-weight: 600; "
                "font-size: 14px; "
                "color: #1971c2;"
            )
        else:
            turn = "White" if self.board.turn else "Black"
            player = "You" if ((self.board.turn == chess.WHITE) == self.user_plays_white) else "AI"
            status = f"Turn: {turn} ({player})"
            self.status_label.setStyleSheet(
                "background-color: #f8f9fa; "
                "border: none; "
                "border-radius: 8px; "
                "padding: 16px; "
                "font-weight: 500; "
                "font-size: 14px; "
                "color: #212529;"
            )
        
        self.status_label.setText(status)
    
    def update_move_history(self):
        """Update the FEN display."""
        # Display current board position in FEN notation
        fen = self.board.fen()
        self.move_list.setPlainText(fen)
        self.move_list.verticalScrollBar().setValue(
            self.move_list.verticalScrollBar().maximum()
        )
    
    def update_notation(self):
        """Update the PGN display."""
        if not self.move_history:
            self.notation_text.clear()
            return
        
        # Build PGN format
        pgn_lines = []
        pgn_lines.append('[Event "Chess Game"]')
        pgn_lines.append('[Site "Local"]')
        pgn_lines.append('[Date "2025"]')
        pgn_lines.append('[Round "-"]')
        pgn_lines.append('[White "Player"]')
        pgn_lines.append('[Black "AI"]')
        pgn_lines.append('[Result "*"]')
        pgn_lines.append('')
        
        # Format moves in PGN style
        move_text = ""
        for i, (move, san, _) in enumerate(self.move_history):
            if i % 2 == 0:
                move_text += f"{i//2 + 1}. {san} "
            else:
                move_text += f"{san} "
        
        pgn_lines.append(move_text.strip())
        pgn_content = '\n'.join(pgn_lines)
        
        self.notation_text.setPlainText(pgn_content)
        self.notation_text.verticalScrollBar().setValue(
            self.notation_text.verticalScrollBar().maximum()
        )
    
    def update_buttons(self):
        """Update button states."""
        game_over = (self.board.is_checkmate() or self.board.is_stalemate() or
                    self.board.is_insufficient_material() or self.board.is_seventyfive_moves() or
                    self.board.is_fivefold_repetition())
        
        can_undo = len(self.board_history) > 1
        self.undo_button.setEnabled(can_undo and not game_over)
    
    def is_user_turn(self) -> bool:
        """Check if it's the user's turn."""
        return (self.board.turn == chess.WHITE) == self.user_plays_white
    
    def handle_user_move(self, move: chess.Move):
        """Handle a move made by the user."""
        if not self.is_user_turn():
            QMessageBox.warning(self, "Not Your Turn", "It's the AI's turn to move.")
            return
        
        if move not in self.board.legal_moves:
            QMessageBox.warning(self, "Invalid Move", "That move is not legal.")
            return
        
        self.make_move(move, is_user=True)
        
        # Check for game end
        if not self.is_game_over():
            # Schedule AI move
            QTimer.singleShot(500, self.make_ai_move)
    
    def make_move(self, move: chess.Move, is_user: bool):
        """Make a move on the board."""
        # Save board state for undo
        board_copy = self.board.copy()
        self.board_history.append(board_copy)
        
        # Get SAN notation BEFORE pushing the move (required because board state changes after push)
        try:
            san = self.board.san(move)
        except:
            san = move.uci()
        
        # Make the move
        self.board.push(move)
        self.move_count += 1
        
        # Add to history
        self.move_history.append((move, san, is_user))
        
        # Update last move highlight
        self.board_widget.set_last_move(move.from_square, move.to_square)
        
        # Update display
        self.update_display()
        
        # Check for game end
        if self.is_game_over():
            self.show_game_end_dialog()
    
    def make_ai_move(self):
        """Make the AI's move."""
        if not self.is_user_turn() and not self.is_game_over():
            self.status_label.setText("AI is thinking...")
            self.status_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px; color: blue;")
            QApplication.processEvents()
            
            try:
                move = get_model_move(self.model, self.board, self.move_count)
                self.make_move(move, is_user=False)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"AI move failed: {e}")
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return (self.board.is_checkmate() or self.board.is_stalemate() or
                self.board.is_insufficient_material() or self.board.is_seventyfive_moves() or
                self.board.is_fivefold_repetition())
    
    def show_game_end_dialog(self):
        """Show dialog when game ends."""
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn else "White"
            msg = f"Checkmate! {winner} wins!"
        elif self.board.is_stalemate():
            msg = "Stalemate! The game is a draw."
        elif self.board.is_insufficient_material():
            msg = "Draw by insufficient material."
        elif self.board.is_seventyfive_moves():
            msg = "Draw by 75-move rule."
        elif self.board.is_fivefold_repetition():
            msg = "Draw by fivefold repetition."
        else:
            return
        
        QMessageBox.information(self, "Game Over", msg)
    
    def undo_move(self):
        """Undo the last move(s)."""
        # Can't undo if we're at the initial position
        if len(self.board_history) <= 1 or not self.move_history:
            return
        
        # Always undo just one move at a time
        # If it's currently the user's turn, the last move was AI's - undo it
        # If it's currently AI's turn, the last move was user's - undo it
        current_turn_is_user = self.is_user_turn()
        last_move_was_user = self.move_history[-1][2]
        
        # Safety check: ensure we have enough history
        if len(self.board_history) < 2:
            return
        
        # Undo the last move
        self.board_history.pop()
        self.move_history.pop()
        self.move_count = max(0, self.move_count - 1)
        
        # Restore board state from history
        self.board = self.board_history[-1].copy()
        
        # Update last move highlight
        if len(self.move_history) >= 1:
            last_move = self.move_history[-1][0]
            self.board_widget.set_last_move(last_move.from_square, last_move.to_square)
        else:
            self.board_widget.set_last_move(None, None)
        
        self.update_display()
        
        # If after undo it's now AI's turn and game isn't over, trigger AI move
        if not self.is_user_turn() and not self.is_game_over():
            QTimer.singleShot(500, self.make_ai_move)
    
    def ask_color_selection(self):
        """Show color selection dialog and update user preference."""
        dialog = ColorSelectionDialog()
        result = dialog.exec()
        # Dialog closes automatically after selection
        if result:
            self.user_plays_white = dialog.user_plays_white
            # Auto-flip board when playing as black
            if not self.user_plays_white:
                self.board_widget.flipped = True
            else:
                self.board_widget.flipped = False
            self.update_display()
            # If AI goes first, make its move
            if not self.user_plays_white and not self.is_game_over():
                QTimer.singleShot(500, self.make_ai_move)
    
    def reset_game(self):
        """Reset the game to initial position."""
        reply = QMessageBox.question(self, "Reset Game", "Are you sure you want to reset the game?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.board = chess.Board()
            self.move_count = 0
            self.move_history = []
            self.board_history = [chess.Board()]
            self.board_widget.set_last_move(None, None)
            # Ask for color selection when resetting
            self.ask_color_selection()
            self.update_display()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python user_play.py <model_path> [--use-conv True/False]")
        print("Example: python user_play.py policy_net_extensive.pt")
        print("Example: python user_play.py policy_net_default.pt --use-conv True")
        print("\nNote: PyQt6 is required. Install with: pip install PyQt6")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Parse use_conv argument (default to True to match training_test.py)
    use_conv = True
    if len(sys.argv) > 2:
        if '--use-conv' in sys.argv:
            idx = sys.argv.index('--use-conv')
            if idx + 1 < len(sys.argv):
                use_conv = sys.argv[idx + 1].lower() in ['true', '1', 'yes']
    
    # Load model
    try:
        model = load_model(model_path, use_conv=use_conv)
    except FileNotFoundError:
        print(f"❌ Error: Model file not found: {model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Apply modern 2025 corporate aesthetic stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QWidget {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 14px;
            color: #1a1a1a;
        }
        QLabel {
            color: #1a1a1a;
        }
        QTextEdit {
            selection-background-color: #cfe2ff;
        }
        QPushButton {
            background-color: #ffffff;
            border: 1px solid #dee2e6;
            border-top: 1px solid #ffffff;
            border-left: 1px solid #ffffff;
            border-right: 1px solid #adb5bd;
            border-bottom: 2px solid #868e96;
            border-radius: 8px;
            padding: 12px 20px;
            font-size: 14px;
            font-weight: 500;
            color: #1a1a1a;
        }
        QPushButton:hover {
            background-color: #f8f9fa;
            border-right: 2px solid #868e96;
            border-bottom: 3px solid #6c757d;
        }
        QPushButton:pressed {
            background-color: #e9ecef;
            border-top: 2px solid #adb5bd;
            border-left: 2px solid #adb5bd;
            border-right: 1px solid #ffffff;
            border-bottom: 1px solid #ffffff;
        }
        QTextEdit {
            background-color: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 12px;
            font-size: 14px;
            color: #1a1a1a;
        }
        QTextEdit:focus {
            border-color: #0d6efd;
        }
        QDialog {
            background-color: #ffffff;
        }
    """)
    
    # Create main window first (with default color, will be updated)
    window = ChessGameWindow(model, user_plays_white=True)
    window.show()
    
    # Process events to ensure window renders
    QApplication.processEvents()
    
    # Show color selection dialog after window is visible
    dialog = ColorSelectionDialog()
    result = dialog.exec()
    # Dialog closes automatically when accept() is called
    if not result:
        window.close()
        sys.exit(0)
    
    # Get the selected color and update window
    user_plays_white = dialog.user_plays_white
    window.user_plays_white = user_plays_white
    
    # Auto-flip board when playing as black
    if not user_plays_white:
        window.board_widget.flipped = True
    else:
        window.board_widget.flipped = False
    
    window.update_display()
    
    # If AI goes first, make its move
    if not user_plays_white:
        QTimer.singleShot(500, window.make_ai_move)
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
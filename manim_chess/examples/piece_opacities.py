from manim import *
from chess_board import ChessBoard
import numpy as np

class Chess(Scene):
    def construct(self):
        piece_opacities = np.random.rand(8, 8)
        board = ChessBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        board.set_piece_opacities(piece_opacities)
        self.add(board.move_to(ORIGIN))

 
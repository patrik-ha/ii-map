from manim import *
from chess_board import ChessBoard
import numpy as np

class Chess(Scene):
    def construct(self):
        board = ChessBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

        arrows = [(0, 0, 1, 1), (1, 1, 1, 1)]
        for arrow in arrows:
            board.add_arrow(*arrow)
        self.add(board.move_to(ORIGIN))

 
from manim import *
from chess_board import ChessBoard
import numpy as np

class Chess(Scene):
    def construct(self):
        board = ChessBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

        highlights = np.random.randint(0, 8, size=(6, 2))
        for (i, j) in highlights:
            board.add_highlight(i, j, BLUE)
        self.add(board.move_to(ORIGIN))

 
from manim import *
from chess_board import ChessBoard

class Chess(Scene):
    def construct(self):
        board = ChessBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        smaller_board = ChessBoard("rqkr/pppp/4/PPPP/RQKR w - - 0 1").next_to(board, LEFT)
        self.add(Group(board, smaller_board).move_to(ORIGIN))


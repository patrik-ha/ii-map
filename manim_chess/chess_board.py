import os
from typing import Tuple

import numpy as np
from manim import *

SQUARE_Z = 0
HIGHLIGHT_Z = 1
PIECE_Z = 2

class ChessBoard(Mobject):
    def __init__(self, fen, **kwargs):
        super().__init__(**kwargs)
        self.fen_rows, self.dims, self.turn = self._read_fen(fen)
        self.icons = [[None for _ in range(self.dims[1])] for _ in range(self.dims[0])]
        self.squares = [[None for _ in range(self.dims[1])] for _ in range(self.dims[0])]
        self.draw_empty_board()
        self.draw_pieces()
        
    
    def flip(self):
        self.rotate_about_origin(np.pi)
        for o in self.submobjects:
            if o.name == "ImageMobject":
                o.rotate(np.pi)

    def move_piece(self, i, j, n, m):
        piece_at = self.icons[n][m]
        anims = [None, None]
        if piece_at is not None:
            anims[0] = FadeOut(piece_at)
        
        piece_to_move = self.icons[i][j]

        piece_to_move.generate_target()

        piece_to_move.target.move_to(self.squares[n][m].get_center())
        anims[1] = MoveToTarget(piece_to_move)
        return anims

    def add_arrow(self, i, j, dx, dy, color=None):
        if color is None:
            color = BLUE
        origin_square = self.squares[i][j].get_center()
        target_square = self.squares[i + dx][j + dy].get_center()
        graphical_arrow = Arrow(origin_square, target_square, color=color, stroke_width=12, max_stroke_width_to_length_ratio=6, max_tip_length_to_length_ratio=0.5)
        graphical_arrow.set_z_index(100) 
        circle = Circle(0.45, stroke_width=DEFAULT_STROKE_WIDTH*2, color=color).shift(target_square).set_z_index(100)
        group = Group(circle, graphical_arrow)
        self.add(group)

    def add_highlight(self, i, j, color):
        square = Square(0.999, stroke_width=0, fill_color=color, fill_opacity=0.7)
        square.move_to(self.squares[i][j])
        self.add(square)
        
    def set_piece_opacities(self, opacities: np.ndarray):
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                icon = self.icons[i][j]
                if icon is not None:
                    alpha = opacities[i, j]
                    alpha_mask = np.copy(icon.pixel_array[:, :, 3]) != 0
                    icon.pixel_array[:, :, 3] = int(255 * alpha) * alpha_mask
    
    def set_piece_opacities_square(self, opacities: np.ndarray):
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                icon = self.icons[i][j]
                color = RED if opacities[i, j] < 0.5 else BLUE

                square = Square(0.9, stroke_width=0, fill_color=color, fill_opacity=abs(opacities[i, j] - 0.5) * 2)
                square.move_to(self.squares[i][j])
                self.add(square)

    def _piece_to_icon(self, c):
        prefix = "w" if c.isupper() else "b"
        prefix = prefix if not c.isspace() else ""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        piece_path = os.path.join(dir_path, "png_pieces/{}.png".format(prefix + c.upper()))
        icon = ImageMobject(piece_path)
        icon.set_x(0)
        icon.set_y(0)
        icon.scale(0.27)
        # if c.lower() == "k":
        #     icon.shift(UP * 0.035)
        return icon

    def _read_fen(self, fen):
        if " " in fen:
            fen = fen.split(" ")[0]
        
        rows = fen.split("/")
        self.turn = True if "w" in fen else False


        for i, row in enumerate(rows):
            new_row = ""
            for c in row:
                if c in "123456789":
                    new_row += " " * int(c)
                else:
                    new_row += c
            rows[i] = new_row
        dims = (len(rows), max([len(row) for row in rows]))
        return rows, dims, self.turn

    def draw_pieces(self):
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                if not self.fen_rows[i][j].isspace():
                    icon = self._piece_to_icon(self.fen_rows[i][j]).shift(i * DOWN + j * RIGHT).set_z_index(PIECE_Z)
                    self.icons[i][j] = icon
        
        for row in self.icons:
            for icon in row:
                if icon is not None:
                    self.add(icon)

    def draw_empty_board(self):
        board = []
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                color = "#B58863" if ((i + j + 1) % 2) == 0 else "#F0D9B5"
                square = Square(0.999, stroke_color=BLACK, stroke_width=0)
                square.set_fill(color, 1)
                square.shift(i * DOWN + j * RIGHT)
                square.set_z(SQUARE_Z)
                self.squares[i][j] = square

        for row in self.squares:
            for sq in row:
                self.add(sq)
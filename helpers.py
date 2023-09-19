import os

import chess
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from manim import *

from manim_chess.chess_board import ChessBoard
from processing.leela_board import LeelaBoard


def process_puzzles_to_fen_move_pairs(puzzles):
    results = []
    for i, row in puzzles.iterrows():
        fen, moves = row["FEN"], row["Moves"]
        pre_move, target_move = moves.split(" ")
        board = chess.Board(fen)
        board.push(chess.Move.from_uci(pre_move))
        results.append((board.fen(), target_move))
    return results


def get_model_and_masker():
    model = keras.models.load_model("ii_map_models/complete_rep_0")
    masker = keras.Model(model.input, model.get_layer("heatmap").output)
    return model, masker

def fen_to_features(fen: str) -> np.ndarray:
    board = LeelaBoard()
    board.pc_board.set_fen(fen)
    board._lcz_push()

    return np.expand_dims(board.lcz_features()[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -8, -7, -6, -5, -4, -3, -2, -1]], 0)

def mask_fen(fen: str, masker, move=None) -> np.ndarray:
    board = LeelaBoard()
    board.pc_board.set_fen(fen)
    board._lcz_push()

    planes = np.expand_dims(board.lcz_features()[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -8, -7, -6, -5, -4, -3, -2, -1]], 0)
    base_masked = np.squeeze(masker(planes).numpy())
    max_masked = np.squeeze(masker(planes).numpy()).max(axis=0)
    board_planes = planes[0, :12]
    empty_squares = board_planes.sum(axis=0) == 0
    full_squares = board_planes.sum(axis=0) > 0


    # For squares that have no pieces: show the importance of the presence of *any* piece
    empty_mask = empty_squares * max_masked
    # For squares that have a piece: show the importance of the piece that is there
    full_mask = (base_masked * board_planes).sum(axis=0)
    masked_position = empty_mask + full_mask
    print(masked_position.shape)

    if board.turn:
        masked_position = np.flipud(masked_position)
    manim_board = ChessBoard(fen)
    manim_board.set_piece_opacities_square(masked_position)

    if not board.turn:
        manim_board.flip()

    if move is not None:
        move = chess.Move.from_uci(move)
        i, j = chess.square_rank(move.from_square), chess.square_file(move.from_square)
        l, k = chess.square_rank(move.to_square), chess.square_file(move.to_square)
        i = (7 - i)
        l = (7 - l)

        dx, dy = l - i, k - j
        
        manim_board.add_arrow(i, j, dx, dy, color=WHITE)

    return board, manim_board, masked_position, planes, base_masked
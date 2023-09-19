# TODO: define concept functions taking a chess-board object, and returning a boolean value
# Could probably deal with the bitboards directly

import chess

import numpy as np

def in_check(board: chess.Board):
    return board.is_check()

def random(board: chess.Board):
    return np.random.random() > 0.5

def threat_my_queen(board: chess.Board):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        # Might be more than one queen, need to check all of them
        if piece is not None and piece.color == board.turn and piece.piece_type == chess.QUEEN and board.is_attacked_by(not board.turn, square):
            return True
    return False


def threat_opp_queen(board: chess.Board):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        # Might be more than one queen, need to check all of them
        if piece is not None and piece.color != board.turn and piece.piece_type == chess.QUEEN and board.is_attacked_by(board.turn, square):
            return True
    return False


def has_mate_threat(chess: chess.Board):
    for move in chess.legal_moves:
        b = chess.copy()
        b.push(move)
        if b.is_checkmate():
            return True
    return False

def has_contested_open_file(board: chess.Board):
    # First, find positions of own queens and rooks:
    files_to_check = []
    for i in range(8):
        for j in range(8):
            piece_at = board.piece_at(chess.square(j, i))
            if piece_at is None:
                continue
            if piece_at.color == board.turn and (piece_at.piece_type == chess.QUEEN or piece_at.piece_type == chess.ROOK):
                files_to_check.append((j, i))
    if not files_to_check:
        return False
    # If the file, i, only has enemy and own rooks and queens on it, it is open and contested
    enemy_occupant_found = False
    other_found = False
    for file, m in files_to_check:
        
        for k in range(8):
            # Don't check the same square as started with
            if k == m:
                continue

            piece_at = board.piece_at(chess.square(file, k))
            if piece_at is None:
                continue

            # Piece on this file that is not a rook or queen, i.e. file is closed
            if piece_at.piece_type != chess.QUEEN and piece_at != chess.ROOK:
                other_found = True
    
            # Enemy occupant on the file, ok...
            if piece_at.color != board.turn and (piece_at.piece_type == chess.QUEEN or piece_at.piece_type == chess.ROOK):
                enemy_occupant_found = True
    return enemy_occupant_found and not other_found


def has_own_double_pawn(board: chess.Board):
    for i in range(8):
        for j in range(8):
            piece_at = board.piece_at(chess.square(j, i))
            if piece_at is None or piece_at.color != board.turn or piece_at.piece_type != chess.PAWN:
                continue

            for di in range(8):
                if i == di:
                    continue
                neighbour = board.piece_at(chess.square(j, di))
                if neighbour is not None and neighbour.color == board.turn and neighbour.piece_type == chess.PAWN:
                    return True
    return False


def has_opp_double_pawn(board: chess.Board):
    for i in range(8):
        for j in range(8):
            piece_at = board.piece_at(chess.square(j, i))
            if piece_at is None or piece_at.color == board.turn or piece_at.piece_type != chess.PAWN:
                continue

            for di in range(8):
                if i == di:
                    continue
                neighbour = board.piece_at(chess.square(j, di))
                if neighbour is not None and neighbour.color != board.turn and neighbour.piece_type == chess.PAWN:
                    return True
    return False


def material_advantage(position: chess.Board):
    total = sum_of_pieces(position, position.turn) - sum_of_pieces(position, not position.turn)
    return total >= 3



def sum_of_pieces(position: chess.Board, color: bool):
    piece_values = {
        chess.PAWN: 1,
        chess.BISHOP: 2,
        chess.KNIGHT: 2,
        chess.ROOK: 4,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    total = 0
    for i in range(8):
        for j in range(8):
            piece_at = position.piece_at(chess.square(j, i))
            if piece_at is None or piece_at.color != color:
                continue

            total += piece_values[piece_at.piece_type]

    return total
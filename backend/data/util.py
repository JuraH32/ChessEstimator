import chess
import torch


def board_to_array(board):
    """Converts a chess board into a 12-layer 8x8 tensor representing the piece positions."""
    # Board('rnbqkbnr/pppppppp/8/8/8/3P4/PPP1PPPP/RNBQKBNR b KQkq - 0 1')
    board_array = torch.zeros((12, 8, 8), dtype=torch.float32)
    piece_map = board.piece_map()
    # {63: Piece.from_symbol('r'), 62: Piece.from_symbol('n')
    for square, piece in piece_map.items():
        # 1 pawn, 2 knight, 3 bishop, 4 rook, 5 queen, 6 king
        # Piece color is true if white
        # White pieces 0-5, black 6-11
        index = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
        row, col = divmod(square, 8)
        board_array[index, 7 - row, col] = 1
    return board_array
import os

import chess
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.staticfiles import StaticFiles

from app.chess_agent import ChessAgentMaiaMultiple
from app.chess_engine import ChessEngine
from app.rating_model import RatingModel

app = FastAPI()
if os.path.exists("chess-estimator-web-app/dist"):
    app.mount(
        "/",
        StaticFiles(directory="chess-estimator-web-app/dist", html=True),
        name="static",
)


@app.websocket("/ws/play")
async def websocket_endpoint(websocket: WebSocket):
    print("WebSocket connected")
    await websocket.accept()
    try:
        initial_data = await websocket.receive_json()
        clock = initial_data.get("clock")

        if not clock:
            await websocket.send_json({"error": "Clock time must be set before starting the game"})
            await websocket.close()
            return

        chess_engine = ChessEngine()
        rating_model = RatingModel(clock)
        chess_agent = ChessAgentMaiaMultiple(clock)

        while True:
            data = await websocket.receive_json()
            move = data.get("move")
            white_clock = data.get("player_clocks")

            try:
                move_obj = chess_engine.board.parse_san(move)# Parse move in SAN format
            except ValueError:  # Invalid SAN move
                try:
                    move_obj = chess.Move.from_uci(move)  # Parse move in UCI format
                except ValueError:  # Invalid move in both formats
                    await websocket.send_json({"error": "Invalid move format"})
                    continue

            move = move_obj.uci()

            if chess_engine.is_valid_move(move):
                chess_engine.make_move(move)

                rating_estimate = rating_model.estimate_rating(chess_engine.get_board(), white_clock)

                bot_move = chess_agent.get_move(chess_engine.get_board(), rating_estimate).uci()
                chess_engine.make_move(bot_move)

                await websocket.send_json({
                    "player_move": move,
                    "bot_move": bot_move,
                    "rating_estimate": rating_estimate,
                })
            else:
                await websocket.send_json({"error": "Invalid move"})
    except WebSocketDisconnect:
        print("WebSocket disconnected")

# Chess estimator backend and model
## What is it?
This is the code for the backend of the chess estimator project as well as for training the neural network model.
The backend is a FastAPI server that allows you to play one game against a chess bot and then estimates your chess rating based on the moves played.

## How was the model trained?
The model was trained using the Lichess database. The database contains games played by players with known ratings. 
The model was trained to predict the rating of a player based on the moves played in a game. 
It was trained on 4 different time controls: `60+0`(1.1M games), `180+0`(180k), `180+2`(180k), and `300+0`(180k).

## How to run the backend?
1. Create a virtual or conda environment
2. Install the requirements: `pip install -r requirements.txt`
3. Run the server: `uvicorn app.main:app --reload`


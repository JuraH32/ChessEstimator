# Chess estimator
## DEMO
https://chess-estimator.azurewebsites.net/

## What is it?
This is a web application that allows you to play one game against a chess bot and then estimates your chess rating based on the moves played.
The point of it is to see how well you could estimate a player's rating based on the moves of only one game without having to play a lot of them.


## How does it work?
Each move you play is sent to the server which has 2 parts: rating estimator and move picker. The chess estimator is a 
neural network that was trained to estimate the rating of a player based on the moves played. After the chess estimator outputs a rating,
that rating is forwarded to the move picker. The move picker picks a maia bot to use based on the current estimate to try to match the player's
skill.

## Project structure
- `backend` - contains the server code (FastAPI) as well as the neural network code
- `frontend` - contains the React app code

## How to run it?

### Docker
1. Install Docker
2. Run `docker build -t chess-estimator .` to create an image.
3. Run `docker run -p 80:80 chess-estimator` to run the container.

### Local
Run based on instruction from backend and frontend separately.



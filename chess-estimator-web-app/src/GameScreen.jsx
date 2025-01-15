import {useEffect, useState} from 'react';
import {useLocation} from 'react-router-dom';
import {Chessboard} from 'react-chessboard'; // Install: npm install react-chessboard
import {Chess} from 'chess.js'; // Install: npm install chess.js

const GameScreen = () => {
    const location = useLocation();
    const {timeControl} = location.state;
    const [time, increment] = timeControl.split('+').map(Number); // Split and parse the time control directly
    const [estimate, setEstimate] = useState(1600);
    const [connected, setConnected] = useState(false);
    const [gameOver, setGameOver] = useState(false);

    const [whiteTime, setWhiteTime] = useState([time, increment]);
    const [position, setPosition] = useState('start');
    const [isWhiteTurn, setIsWhiteTurn] = useState(true);

    const [chess] = useState(new Chess());
    const [ws, setWs] = useState(null);

    useEffect(() => {
        console.log('Connecting to server...');
        const socket = new WebSocket('/api/ws/play');
        setWs(socket);

        socket.onopen = () => {
            console.log('Connected to server');
            setConnected(true);
            socket.send(JSON.stringify({clock: timeControl}));
        }
        socket.onmessage = (event) => {
            console.log('Received message:', event.data);
            const {bot_move, rating_estimate} = JSON.parse(event.data);
            chess.move(bot_move);
            setTimeout(() => {
                setPosition(chess.fen());
                setEstimate(rating_estimate);
                toggleTurn();
            }, 200); // Adjust delay as needed
        }

        return () => socket.close();
    }, []);

    // Handle time decrement (if increment > 0, deduct increment instead of time)
    const incrementTime = (time) => {
        let [currentTime, currentIncrement] = time;
        if (currentIncrement > 0) {
            currentIncrement -= 1;
        } else {
            currentTime = Math.max(currentTime - 1, 0);
        }
        return [currentTime, currentIncrement];
    }

    const toggleTurn = () => {
        console.log('White turn:', isWhiteTurn);
        if (isWhiteTurn) {
            setWhiteTime((prevWhiteTime) => [prevWhiteTime[0], increment]);
        }

        setIsWhiteTurn((prevIsWhiteTurn) => !prevIsWhiteTurn);
    }

    useEffect(() => {
        if (!isWhiteTurn && ws) {
            // Get UCI move
            const moves = chess.history();
            const lastMove = moves[moves.length - 1];
            console.log('Sending move:', lastMove);
            ws.send(JSON.stringify({move: lastMove, player_clocks: whiteTime}));
        }
    }, [isWhiteTurn]);

    useEffect(() => {
        if (!connected) return;
        const timer = setInterval(() => {
            if (isWhiteTurn) {
                setWhiteTime((prevWhiteTime) => incrementTime(prevWhiteTime));
            }
        }, 1000);

        return () => clearInterval(timer); // Cleanup on component unmount
    }, [isWhiteTurn]); // Depend on isWhiteTurn to trigger timer on turn switch

    useEffect(() => {
        if (whiteTime[0] === 0 && whiteTime[1] === 0) {
            setGameOver(true);
        }
        if (chess.moves().length === 0) {
            setGameOver(true);
        }
    }, [position, whiteTime]);

    // Handle piece movement
    const onDrop = (sourceSquare, targetSquare, piece) => {
        if (!isWhiteTurn) return; // Prevent moving while bot is playing

        const move = chess.move({
            from: sourceSquare,
            to: targetSquare,
            promotion: 'q' // Always promote to queen
        });

        if (move === null) return; // Invalid move

        console.log('Move:', move);

        setPosition(chess.fen()); // Update position after valid move
        toggleTurn(); // Switch turn
    }

    // Format time for display
    const formatTime = (time) => {
        const [currentTime, currentIncrement] = time;
        const minutes = Math.floor(currentTime / 60);
        const seconds = currentTime % 60;
        return `${minutes}:${seconds < 10 ? '0' : ''}${seconds} + ${currentIncrement}`;
    }

    return (
        <div>
            <h1>Time control: {timeControl}</h1>
            {gameOver ? <h1>Estimated ELO: {Math.round(estimate)}</h1> : <>
                <h3>Estimated ELO: {Math.round(estimate)}</h3>
                <div>
                    <h3>Time left: {formatTime(whiteTime)}</h3>
                </div>
                {connected ?
                    <Chessboard
                        position={position}
                        onPieceDrop={onDrop}
                        boardOrientation={'white'}
                        animationDuration={200}
                        customDarkSquareStyle={{backgroundColor: '#769656'}}
                        customLightSquareStyle={{backgroundColor: '#eeeed2'}}

                    /> : <h2>Connecting to server...</h2>}
                <div>
                </div>
            </>}
        </div>
    );
};

export default GameScreen;

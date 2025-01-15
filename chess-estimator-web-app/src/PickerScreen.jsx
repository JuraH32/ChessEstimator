// src/PickerScreen.jsx
import {useState} from 'react';
import { useNavigate } from 'react-router-dom';

const PickerScreen = () => {
    const [timeControl, setTimeControl] = useState("60+0")
    const navigate = useNavigate();

    const startGame = () => {
        navigate('/game', { state: { timeControl } });
    };

    return (
        <div>
            <h1>Choose Your Options</h1>
            <label>
                Select Time control
                <select value={timeControl} onChange={(e) => setTimeControl(e.target.value)}>
                    <option value="60+0">60+0</option>
                    <option value="180+0">180+0</option>
                    <option value="180+2">180+2</option>
                    <option value="300+0">300+0</option>
                </select>
            </label>
            <br/>
            <br/>
            <button onClick={startGame}>Start Game</button>
        </div>
    );
};

export default PickerScreen;

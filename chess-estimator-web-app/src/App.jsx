import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css'
import PickerScreen from "./PickerScreen.jsx";
import GameScreen from "./GameScreen.jsx";

function App() {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<PickerScreen/>}/>
                <Route path="/game" element={<GameScreen/>}/>
            </Routes>
        </Router>
    )
}

export default App

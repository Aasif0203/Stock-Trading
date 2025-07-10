import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import NavBar from './NavBar.jsx'
import './FunctionalPage.css'
// You need to import your components
import Home from './Different pages/Home.jsx' // or wherever these components are located
import Orders from './Different pages/Orders.jsx'
import Funds from './Different pages/Funds.jsx'
import Holdings from './Different pages/Holdings.jsx'
import Positions from './Different pages/Positions.jsx'
// Import other components as needed

export default function MainPage(){
  return(
    <div className='mainpage' style={{flexGrow:2}}>
      <Router>
        <NavBar />
        <Routes>
          <Route exact path="/" element={<Home />} />
          <Route path="/Orders" element={<Orders />} />
          <Route path="/Funds" element={<Funds />} />
          <Route path="/Holdings" element={<Holdings />} />
          <Route path="/Positions" element={<Positions />} />
        </Routes>
      </Router>
    </div>
  )
}
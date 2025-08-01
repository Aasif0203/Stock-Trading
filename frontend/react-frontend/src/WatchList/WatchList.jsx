import './WatchList.css';
import {watchlist} from '../data.jsx';
import { useState } from 'react';
import ArrowDropUpIcon from '@mui/icons-material/ArrowDropUp';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import Tooltip from '@mui/material/Tooltip';
import Button from '@mui/material/Button';

export default function WatchList(){
  return (
    <div id="WatchListTab" style={{flexGrow:1}} >
      <ul className='list' >
        {
          watchlist.map((stock, idx) => (
            <WatchListItem stock={stock} key={idx}/>
          ))
        }
      </ul>
    </div>
  )
}
function WatchListItem({stock}){
  let [showMouse, SetShowMouse] = useState(false);

  let handleMouseEvent = (e)=>{
    SetShowMouse(!showMouse);
  }

  return (
    <li className='watchListItem' style={{listStyleType:'none', position:'relative'}} onMouseEnter={handleMouseEvent} onMouseLeave={handleMouseEvent}>
      <p className='stkname'>{stock.name} </p>
      <p style={{opacity:'0.5'}} className={stock.isDown? "stockDown":"stockUp"}>{stock.percent} </p>
      {stock.isDown? <ArrowDropDownIcon style={{color: 'red'}} />:<ArrowDropUpIcon style={{color: 'green'}}/>}
      <p style={{fontSize:'13px', fontFamily:'sans-serif'}}>{stock.price} </p>
      {showMouse && <WatchListActions uid={stock.name}/> }
    </li>
  )
}

function WatchListActions({uid}){
  return (
    <div style={{ position: 'absolute',right:130, zIndex: 10 }}>
      <Tooltip title={`Buy ${uid} stock`} placement="top">
        <Button variant="contained" color='primary' size="small" style={{margin:'2px'}}>
          Buy
        </Button>
      </Tooltip>
      <Tooltip title={`Sell ${uid} stock`} placement="top">
        <Button variant="contained" color='error' size="small" style={{margin:'2px'}}>
          Sell
        </Button>
      </Tooltip>
    </div>
  )
}
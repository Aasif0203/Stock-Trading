import Button from '@mui/material/Button';
import { useState } from 'react';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';
import ArrowDropUpIcon from '@mui/icons-material/ArrowDropUp';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import Tooltip from '@mui/material/Tooltip';

export default function WatchListItem({stock,onBuyTriggered,onSellTriggered}){
  let [showMouse, SetShowMouse] = useState(false);

  let handleMouseEvent = (e)=>{
    SetShowMouse(!showMouse);
  }
  return (
    <li className='watchListItem' style={{listStyleType:'none', position:'relative'}} onMouseEnter={handleMouseEvent} onMouseLeave={handleMouseEvent}>
      <p className='stkname'>{stock.name} </p>
      <p style={{opacity:'0.7'}} className={stock.isLoss? "stockDown":"stockUp"}>{stock.percentChange}% </p>
      {stock.isLoss? <ArrowDropDownIcon style={{color: 'red'}} />:<ArrowDropUpIcon style={{color: 'green'}}/>}
      <p style={{fontSize:'13px', fontFamily:'sans-serif'}}>{stock.currentPrice} </p>
      {showMouse && <WatchListActions stockname={stock.name} onBuyTriggered ={onBuyTriggered} onSellTriggered={onSellTriggered} /> }
    </li>
  )
}

function WatchListActions({stockname,onBuyTriggered,onSellTriggered}){
  return (
    <div style={{ position: 'absolute',right:140, zIndex: 10 }}>
      <Tooltip title={`Buy ${stockname} for Market Price`} placement="top">
        <Button variant="contained" color='primary' size="small" style={{margin:'2px'}} onClick={()=>onBuyTriggered(stockname,false)}> 
          Buy
        </Button>
      </Tooltip>
      <Tooltip title={`Sell ${stockname} stock`} placement="top">
        <Button variant="contained" color='error' size="small" style={{margin:'2px'}} onClick={()=>onSellTriggered(stockname)}>
          Sell
        </Button>
      </Tooltip>
      <Tooltip title={`Buy ${stockname} with Limit`} placement="top">
        <Button variant="contained" color='white' size="small" style={{margin:'2px'}} onClick={()=>onBuyTriggered(stockname,true)}>
          <ShoppingCartIcon/>
        </Button>
      </Tooltip>
    </div>
  )
}
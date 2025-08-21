import axios from 'axios';
import Button from '@mui/material/Button';
import ButtonGroup from '@mui/material/ButtonGroup';
import CloseOutlinedIcon from '@mui/icons-material/CloseOutlined';
import { useState } from 'react';

export default function SellWindow({uid,onClose}){
  let [qty,setQTY] = useState(1);
  
  let enterSell = ()=>{
    
    axios.post('http://localhost:3001/addOrder',{
      name:uid,
      mode:"SELL",
      qty:qty,
      pending:false,
    });
    onClose();
  }
  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(238, 2, 2, 0.3)',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      zIndex: 1000
    }}>
      <div className='BuyWindow' style={{
        backgroundColor: 'black',
        padding: '20px',
        borderRadius: '8px',
        minWidth: '300px',
        textAlign: 'center'
      }}>
        <h3>{uid}</h3>
        <label htmlFor="qty">Quantity : </label>
        <input
          type="number"
          id="qty"
          placeholder='Quantity'
          value={qty}
          min="1"
          onChange={e => setQTY(Number(e.target.value))}
        />
        <br/><br/>
        <ButtonGroup variant="outlined"  size='large' aria-label="Large button group">
          <Button color='error' onClick={()=>enterSell()}>Sell</Button>
          <Button onClick={()=>onClose()}><CloseOutlinedIcon/> Cancel</Button>
        </ButtonGroup>
      </div>
    </div>
  )
}
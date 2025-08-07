import axios from 'axios';
import Button from '@mui/material/Button';
import ButtonGroup from '@mui/material/ButtonGroup';
import CloseOutlinedIcon from '@mui/icons-material/CloseOutlined';
import { useState } from 'react';

export default function SellWindow({uid,onClose}){
  let [qty,setQTY] = useState(1);
  let [price,setPrice] = useState(0.1);

  let enterSell = ()=>{
    axios.post('http://localhost:3001/addOrder',{
      name:uid,
      price:price,
      mode:"SELL",
      qty:qty,
      pending:false,
    });
    onClose();
  }
  return (
    <div className='BuyWindow'>
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
      <ButtonGroup variant="outlined"  size='large' aria-label="Large button group">
        <Button color='error' onClick={()=>enterSell()}>Sell</Button>
        <Button onClick={()=>onClose()}><CloseOutlinedIcon/> Cancel</Button>
      </ButtonGroup>
    </div>
  )
}
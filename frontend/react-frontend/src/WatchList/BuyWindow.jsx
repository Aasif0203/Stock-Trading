import axios from 'axios';
import Button from '@mui/material/Button';
import ButtonGroup from '@mui/material/ButtonGroup';
import CloseOutlinedIcon from '@mui/icons-material/CloseOutlined';
import { useState } from 'react';

export default function BuyWindow({uid,actualprice,onClose,pending}){
  let [qty,setQTY] = useState(1);
  let [price,setPrice] = useState(0.1);

  let enterBuy =async ()=>{
    onClose();
    await axios.post('http://localhost:3001/addOrder',{
      name:uid,
      price:pending?price:actualprice ,
      mode:"BUY",
      qty:qty,
      pending:pending,
    })
    
  };
  
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
      <label style={pending? undefined:{display:'none'}} htmlFor="price">Price : </label>
      <input style={pending? undefined:{display:'none'}} type="number" id="price" name="price" min={0.2} step="any" placeholder="Price"
        value={price} onChange={e => setPrice(Number(e.target.value))}
      />
      <br/> <br/>
      <ButtonGroup variant="outlined" color='white' size='large' aria-label="Large button group">
        <Button onClick={()=>enterBuy()}>Buy</Button>
        <Button onClick={()=>onClose()}><CloseOutlinedIcon/> Cancel</Button>
      </ButtonGroup>
    </div>
  )
}
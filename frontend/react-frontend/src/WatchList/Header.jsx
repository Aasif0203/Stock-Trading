import AddIcon from '@mui/icons-material/Add';
import RefreshIcon from '@mui/icons-material/Refresh';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import { useState } from 'react';
import axios from 'axios';

export function Header({loading,handleRefresh}){
  let [ticker,setTicker] = useState('');
  let [error,setError] = useState(null);
  let addStock = async ()=>{
    try {
      let response = await axios.post(`http://localhost:3001/watchlist/${ticker}`);
      setError(null);
      handleRefresh();
    } catch (error) {
      // Access the error message from the server response
      if (error.response && error.response.data && error.response.data.error) {
        setError(error.response.data.error);
      } else {
        setError('An error occurred while adding the stock');
      }
    }
  }
  return (
    <div className="Header">
      <h2 style={{fontSize:'30px', marginTop:'0px',marginBottom:'10px'}}>WatchList</h2>
      <div>
        <Button style={{height:'36px'}} size='small' loading={loading} variant="contained" onClick={()=>handleRefresh()}>
          <b>Refresh</b> <RefreshIcon />
        </Button>
        <form onSubmit={e => { e.preventDefault(); addStock(); }}>
          <TextField onChange={e=> setTicker(e.target.value)} size='small' label="Enter Stock Ticker" variant="filled" focused /> &nbsp;&nbsp;&nbsp;
          <Button variant='contained' type='submit'><AddIcon /> </Button>
        </form>
      </div>
      {error!=null && <p style={{color: 'red'}}>{error}</p>}
    </div>
  )
}
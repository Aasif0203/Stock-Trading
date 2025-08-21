import AddIcon from '@mui/icons-material/Add';
import RefreshIcon from '@mui/icons-material/Refresh';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import { useState } from 'react';
import axios from 'axios';

export function Header({loading,setLoading,handleRefresh}){
  let [ticker,setTicker] = useState('');
  let [error,setError] = useState(null);
  let addStock = async (e)=>{
     e.preventDefault(); 
    
    setLoading(true);
    try {
      await axios.post(`http://localhost:3001/watchlist/${ticker}`);
      await handleRefresh();
      setError(null);
      setTicker('');
      
    } catch (error) {
      const errorMessage = error.response?.data?.error || 'Something went wrong';
      setError(errorMessage);
      setLoading(false);
    }
  }
  return (
    <div className="Header">
      <h2 style={{fontSize:'30px', marginTop:'0px',marginBottom:'10px'}}>WatchList</h2>
      <div>
        <Button style={{height:'36px'}} size='small' loading={loading} variant="contained" onClick={()=>handleRefresh()}>
          <b>Refresh</b> <RefreshIcon />
        </Button>
        <form onSubmit={addStock} >
          <TextField 
            value={ticker}
            onChange={e=> setTicker(e.target.value)} 
            size='small' 
            label="Enter Stock Ticker" 
            variant="filled" 
            focused 
          /> &nbsp;&nbsp;&nbsp;
          <Button variant='contained' type='submit'><AddIcon /> </Button>
        </form>
      </div>
      {error!=null && <p style={{color: 'red'}}>{error}</p>}
    </div>
  )
}
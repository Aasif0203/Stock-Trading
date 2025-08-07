import './WatchList.css';
import axios from 'axios';
import { useState, useEffect } from 'react';

import Button from '@mui/material/Button';
import BuyWindow from './BuyWindow.jsx';
import SellWindow from './SellWindow.jsx';
import WatchListItem from './Item&Action.jsx'
import {DoughnutChart} from './DoughnutChart.jsx';
import { Header } from './Header.jsx';


export default function WatchList(){
  let [buyUID,setBuyUID] = useState(null);
  let [sellUID,setSellUID] = useState(null);
  const [loading, setLoading] = useState(false);
  let [watchlist,setWatchList] = useState([]);
  let [pending,setPending] = useState(false);

  let BuyTriggered = (uid,pending)=>{
    setBuyUID(uid);
    if(pending) setPending(true);
    setSellUID(null);
  }
  let closeBuyWindow = ()=>{
    setBuyUID(null);
    setPending(false);
  }
  let SellTriggered = (uid)=>{
    setBuyUID(null);
    setSellUID(uid);
  }
  let closeSellWindow = ()=>{
    setSellUID(null);
  }
  
  const generateRandomColor = (alpha = 0.2) => {
    const r = Math.floor(Math.random() * 256);
    const g = Math.floor(Math.random() * 256);
    const b = Math.floor(Math.random() * 256);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  };

  // Generate colors based on watchlist length
  const generateColors = (count, alpha) => {
    return Array.from({ length: count }, () => generateRandomColor(alpha));
  };

  const data = {
    labels: watchlist.map((data)=>data.name),
    datasets: [
      {
        label: 'Price',
        data: watchlist.map((data)=>data.currentPrice),
        backgroundColor: generateColors(watchlist.length, 0.5),
        borderColor: generateColors(watchlist.length, 1),
        borderWidth: 1.5,
      },
    ],
  };

  // Fetch multiple stocks concurrently
  const fetchMultipleStocks = async () => {
    setLoading(true);
    try {
      await axios.put('http://localhost:3001/watchlist');
      const res = await axios.get('http://localhost:3001/watchlist');
      setWatchList(res.data);
    } catch (error) {
      console.error('Error fetching watchlist:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMultipleStocks();
  }, []);


  return (
    <div id="WatchListTab" style={{flexGrow:1}} >
      <br/>
      <Header loading={loading} handleRefresh={fetchMultipleStocks} />
      <ul >
        {
          watchlist.map((stock, idx) => (
            <WatchListItem stock={stock} key={idx} onBuyTriggered={BuyTriggered} onSellTriggered={SellTriggered} />
          ))
        }
      </ul>
      {buyUID && <BuyWindow uid={buyUID} actualprice={watchlist.find(stock => stock.name === buyUID)?.currentPrice} pending={pending} onClose={closeBuyWindow}/>}
      {sellUID && <SellWindow uid={sellUID} onClose={closeSellWindow}/>} <br/><hr/>
      <DoughnutChart data={data} />
    </div>
  )
}

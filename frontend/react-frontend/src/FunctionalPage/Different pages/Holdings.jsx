import React, { useState, useEffect } from "react";
import { VerticalGraph } from "./VerticalGraph";
import axios from "axios";
import Tooltip from "@mui/material/Tooltip";
import Button from "@mui/material/Button";
import SellWindow from "../../WatchList/SellWindow.jsx";

export default function Holdings(){
  const [allHoldings, setAllHoldings] = useState([]);
  const [sellUID, setSellUID] = useState(null);

  const handleSellClick = (stockName) => {
    setSellUID(stockName);
  };

  const closeSellWindow = () => {
    setSellUID(null);
  };

  useEffect(() => {
    axios.get("http://localhost:3001/allHoldings").then((res) => {
      console.log(res.data);  
      setAllHoldings(res.data);
    });
  });

  // const labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July'];
  const labels = allHoldings.map((subArray) => subArray["name"]);

  const data = {
    labels,
    datasets: [
      {
        label: "Stock Price",
        data: allHoldings.map((stock) => stock.price),
        backgroundColor: "rgba(185, 227, 36, 0.5)",
      },
    ],
  };

  // export const data = {
  //   labels,
  //   datasets: [
  // {
  //   label: 'Dataset 1',
  //   data: labels.map(() => faker.datatype.number({ min: 0, max: 1000 })),
  //   backgroundColor: 'rgba(255, 99, 132, 0.5)',
  // },
  //     {
  //       label: 'Dataset 2',
  //       data: labels.map(() => faker.datatype.number({ min: 0, max: 1000 })),
  //       backgroundColor: 'rgba(53, 162, 235, 0.5)',
  //     },
  //   ],
  // };

  const totalInvestment = allHoldings.reduce((sum, stock) => sum + stock.invested, 0);
  const totalCurrentValue = allHoldings.reduce((sum, stock) => sum + stock.price * stock.qty, 0);
  const totalPL = totalCurrentValue - totalInvestment;
  const totalPLPercent = totalInvestment > 0 ? ((totalPL / totalInvestment) * 100).toFixed(2) : "0.00";

  return (
    <>
      <h3 className="title">Holdings ({allHoldings.length})</h3>

      <div className="order-table">
        <table>
          <thead>
            <tr>
              <th style={{borderRight:'1px solid grey'}}>Instrument</th>
              <th>Qty.</th>
              <th>Invested</th>
              <th>LTP</th>
              <th>Cur. val</th>
              <th>P&L</th>
              <th>Net chg.</th>
              <th>Day chg.</th>
            </tr>
          </thead>
          <tbody>
            {allHoldings.map((stock, index) => {
              const curValue = stock.price * stock.qty;
              const isProfit = curValue - stock.invested >= 0.0;
              const profClass = isProfit ? "profit" : "loss";
              const dayClass = stock.isLoss ? "loss" : "profit";

              return (
                <tr key={index}>
                  <td style={{borderRight:'1px solid grey'}}>{stock.name}</td>
                  <td>{stock.qty}</td>
                  <td>{stock.invested.toFixed(2)}$</td>
                  <td>{stock.price.toFixed(2)}$</td>
                  <td>{curValue.toFixed(2)}</td>
                  <td className={profClass}>
                    {(curValue - stock.invested).toFixed(2)}
                  </td>
                  <td className={profClass}>{stock.net}</td>
                  <td className={dayClass}>{stock.day}</td>
                  <td>
                    <Tooltip title={`Sell ${stock.name} stock`} placement="top">
                      <Button variant="contained" color='error' size="small" style={{margin:'2px'}} onClick={()=>handleSellClick(stock.name)}>
                        Sell
                      </Button>
                    </Tooltip>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="HoldingBox">
        <div>
          <h2>
            {totalInvestment.toFixed(2)}
          </h2>
          <p>Total investment</p>
        </div>
        <div className="col">
          <h2>
            {totalCurrentValue.toFixed(2)}
          </h2>
          <p>Current value</p>
        </div>
        <div className="col">
          <h2>
            {totalPL.toFixed(2)} ({totalPL >= 0 ? "+" : ""}{totalPLPercent}%)
          </h2>
          <p>P&L</p>
        </div>
      </div>
      <VerticalGraph data={data} />
      {sellUID && <SellWindow uid={sellUID} onClose={closeSellWindow}/>}
    </>
  );
};


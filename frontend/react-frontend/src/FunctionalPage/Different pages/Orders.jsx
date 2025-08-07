import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import OrderItem from "./OrderItem";
import './Orders.css';
import axios from "axios";

export default function Orders(){
  let [orders,setOrders] = useState([]);
  let [empty,setEmpty] = useState(true);

  useEffect(()=>{
    fetchorders();
  });
  let fetchorders = async ()=>{
    await axios.get('http://localhost:3001/allOrders').then((res)=>{
      setOrders(res.data);
      if(res.data.length !== 0) setEmpty(false);
      else setEmpty(true);
    }).catch(()=>{
      setEmpty(true);
    })
  }

  return (
    <div className="orders">
      {empty && <EmptyOrder />}
      {!empty && (
        <div className="Orders">
          {orders
            .map((data, idx) => (
              <div
                key={data._id}
                style={{
                  opacity: 0,
                  transform: "translateY(-20px)",
                  animation: `fadeInUp 0.5s ease ${(orders.length- idx) * 0.1 + 0.2}s forwards`
                }}
              >
                <OrderItem orders={data} />
              </div>
            )).reverse()}
        </div>
      )}
      <style>
        {`
          @keyframes fadeInUp {
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }
        `}
      </style>
    </div>
  );
};
function EmptyOrder(){
  return (
    <div className="no-orders">
      <h2>You haven't placed any orders today</h2>

      <Link to={"/"} className="btn">
        Get started
      </Link>
    </div>
  );
}

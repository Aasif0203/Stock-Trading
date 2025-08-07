import CheckCircleOutlinedIcon from '@mui/icons-material/CheckCircleOutlined';
import PendingOutlinedIcon from '@mui/icons-material/PendingOutlined';

export default function OrderItem({orders}){
  return (
    <div className={orders.mode === 'SELL' ? "sellbox" : orders.isPending? "pendingbox":"buybox"}>
      <span>
        <h2>{orders.name}</h2>
      </span>
      <span ><h4 style={{margin:'10px'}} className='qty'><b>Quantity :</b> {orders.qty}</h4> </span>
      <span ><h4 className='price'><b>Each Stock Price :</b> ${orders.price} </h4></span>
      <span>
        <h4 > {orders.isPending ? <><b style={{color:'orange'}}>Pending </b><PendingOutlinedIcon /></> : <><b style={{color:'green'}}>Complete </b><CheckCircleOutlinedIcon /></>}</h4>
      </span>
      <span>
        <span><p style={{margin:'3px'}}><b>{orders.mode}</b> On</p> </span>
        <span><p style={{margin:'3px'}}>{orders.date}</p><p style={{margin:'3px'}}>{orders.time}</p> </span>
      </span>
      
    </div>
  )
}
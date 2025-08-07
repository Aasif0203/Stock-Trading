const {order} = require('../models/OrderModel');
const express = require('express');
const router = express.Router();
const {pending} = require('../models/PendingModel');

router.post('/addOrder', async (req,res)=>{
  let newOrder = new order({
    name: req.body.name,
    qty:req.body.qty,
    price: req.body.price,
    mode: req.body.mode,
    isPending: req.body.pending,
    day: new Date().toDateString(),
    time: new Date().toTimeString().split(' ')[0],
  })
  newOrder.save().then((r)=> console.log(r.data));
  if (req.body.pending) {
      let newPending = new pending({
        name: req.body.name,
        price: req.body.price
      });
      await newPending.save();
    }
});
router.get('/allOrders', async (req,res)=>{
  res.json(await order.find({}));
})

module.exports = router;
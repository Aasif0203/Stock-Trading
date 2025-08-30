const {order} = require('../models/OrderModel');
const {holding, addHolding, sellHolding} = require('./holdings')
const { requireAuth } = require('../util/AuthMiddleware');
const express = require('express');
const router = express.Router();
const {pending} = require('../models/PendingModel');

router.post('/addOrder', requireAuth, async (req, res) => {
  const { name, qty, price, mode, isPending } = req.body;
  
  // Always create an order record first
  const newOrder = new order({
    userId: req.user._id,
    name: name,
    qty: qty,
    price: price,
    mode: mode.toUpperCase(),
    isPending: isPending || false,
    day: new Date().toDateString(),
    time: new Date().toTimeString().split(' ')[0],
  });
  
  const savedOrder = await newOrder.save();
  
  if (isPending) {
    // For limit orders, also save to pending collection for tracking
    const newPending = new pending({
      userId: req.user._id,
      name: name,
      targetPrice: price,
      qty: qty,
      mode: mode.toUpperCase()
    });
    await newPending.save();
    
    res.json({
      message: `Limit order for ${name} placed. Will ${mode.toLowerCase()} ${qty} shares when price reaches $${price}`,
      order: savedOrder,
      pendingOrder: newPending
    });
  } else {
    // Execute order immediately (market order)
    let result;
    
    if (mode.toUpperCase() === 'BUY') {
      result = await addHolding(name, qty, price, req.user._id);
    } else if (mode.toUpperCase() === 'SELL') {
      result = await sellHolding(name, qty, price, req.user._id);
    }
    
    // Send response based on order type
    if (mode.toUpperCase() === 'BUY') {
      res.json({
        message: `Successfully bought ${qty} shares of ${name}`,
        order: savedOrder,
        totalCost: result.totalCost,
        remainingFunds: result.remainingFunds
      });
    } else if (mode.toUpperCase() === 'SELL') {
      res.json({
        message: result.message,
        order: savedOrder,
        totalReceived: result.totalValue,
        newFunds: result.newFunds
      });
    }
  }
});
router.get('/allOrders', requireAuth, async (req,res)=>{
  // Only get orders for current user
  res.json(await order.find({ userId: req.user._id }));
})

module.exports = router;
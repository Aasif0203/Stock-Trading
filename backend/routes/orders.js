const {order} = require('../models/OrderModel');
const {holding, addHolding, sellHolding} = require('./holdings')
const express = require('express');
const router = express.Router();
const {pending} = require('../models/PendingModel');

router.post('/addOrder', async (req, res) => {
  try {
    const { name, qty, price, mode, pending } = req.body;
    
    // Validation
    if (!name || !qty || !mode) {
      throw new Error('Name, quantity, and mode are required');
    }
    
    if (qty <= 0) {
      throw new Error('Quantity must be positive');
    }
    
    if (!['BUY', 'SELL'].includes(mode.toUpperCase())) {
      throw new Error('Mode must be either BUY or SELL');
    }
    
    // For BUY orders, price is mandatory
    if (mode.toUpperCase() === 'BUY' && (!price || price <= 0)) {
      throw new Error('Price is required for BUY orders and must be positive');
    }
    
    // For SELL orders, price is optional (will get from Finnhub if not provided)
    if (mode.toUpperCase() === 'SELL' && price && price <= 0) {
      throw new Error('If price is provided for SELL orders, it must be positive');
    }
    
    // Execute order immediately
    let result;
    let actualPrice = price;
    
    if (mode.toUpperCase() === 'BUY') {
      result = await addHolding(name, qty, price);
      actualPrice = price;
    } else if (mode.toUpperCase() === 'SELL') {
      result = await sellHolding(name, qty, price); // sellHolding will get real-time price if price is null
      actualPrice = result.soldPrice; // Use the actual price from sellHolding
    }
    
    // Create order record with actual price used
    const newOrder = new order({
      name: name,
      qty: qty,
      price: actualPrice,
      mode: mode.toUpperCase(),
      isPending: pending || false,
      day: new Date().toDateString(),
      time: new Date().toTimeString().split(' ')[0],
    });
    
    const savedOrder = await newOrder.save();
    console.log('Order saved:', savedOrder._id);
    
    if (pending) {
      // Handle pending orders
      const newPending = new pending({
        name: name,
        price: actualPrice,
        mode: mode.toUpperCase(),
        qty: qty
      });
      await newPending.save();
      
      res.json({
        message: `${mode.toUpperCase()} order for ${name} added to pending orders`,
        order: savedOrder
      });
    } else {
      // Send response based on order type
      if (mode.toUpperCase() === 'BUY') {
        res.json({
          message: `Successfully bought ${qty} shares of ${name}`,
          order: savedOrder,
          holding: result
        });
      } else if (mode.toUpperCase() === 'SELL') {
        res.json({
          message: result.message,
          order: savedOrder,
          soldDetails: result
        });
      }
    }
    
  } catch (error) {
    console.error('Error processing order:', error.message);
    res.status(400).json({ error: error.message });
  }
});
router.get('/allOrders', async (req,res)=>{
  res.json(await order.find({}));
})

module.exports = router;
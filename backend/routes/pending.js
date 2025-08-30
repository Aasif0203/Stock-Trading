const express = require('express');
const {pending} = require('../models/PendingModel');
const {addHolding, sellHolding} = require('./holdings');
const { order } = require('../models/OrderModel');
const { requireAuth } = require('../util/AuthMiddleware');
const router = express.Router();
const {getStockQuote} = require('../services/finnhub');

// Get user's pending orders
router.get('/', requireAuth, async (req, res) => {
  const userPendingOrders = await pending.find({ userId: req.user._id });
  res.json(userPendingOrders);
});

const processPendingOrders = async () => {
  console.log('Starting to process pending orders...');
  
  // Get all pending orders
  const pendingOrders = await pending.find({});
  
  if (pendingOrders.length === 0) {
    console.log('No pending orders to process');
    return;
  }

  console.log(`Processing ${pendingOrders.length} pending orders...`);

  for (const pendingOrder of pendingOrders) {
    const stockData = await getStockQuote(pendingOrder.name);
    const currentPrice = stockData.c; // Use raw Finnhub format
    const targetPrice = pendingOrder.targetPrice;
    const mode = pendingOrder.mode;

    console.log(`Checking ${pendingOrder.name}: Current=${currentPrice}, Target=${targetPrice}, Mode=${mode}`);

    let shouldExecute = false;

    // Determine if order should be executed
    if (mode === 'BUY' && currentPrice <= targetPrice) {
      shouldExecute = true; // Buy when price drops to target or below
    } else if (mode === 'SELL' && currentPrice >= targetPrice) {
      shouldExecute = true; // Sell when price rises to target or above
    }

    if (shouldExecute) {
      console.log(`Executing ${mode} order for ${pendingOrder.name} at ${currentPrice}`);
      
      let result;
      if (mode === 'BUY') {
        result = await addHolding(pendingOrder.name, pendingOrder.qty, currentPrice, pendingOrder.userId);
      } else if (mode === 'SELL') {
        result = await sellHolding(pendingOrder.name, pendingOrder.qty, currentPrice, pendingOrder.userId);
      }

      // Update the original order to mark as executed
      await order.updateOne(
        { 
          userId: pendingOrder.userId,
          name: pendingOrder.name, 
          isPending: true,
          mode: mode
        },
        { 
          isPending: false,
          price: currentPrice // Update with actual execution price
        }
      );
      
      // Remove from pending collection (order completed)
      await pending.deleteOne({ _id: pendingOrder._id });
      
      console.log(`✓ Executed ${mode} order: ${pendingOrder.qty} shares of ${pendingOrder.name} at $${currentPrice}`);
    }
  }

  console.log('Finished processing pending orders');
};

module.exports = {
  pending: router,
  processPendingOrders
};
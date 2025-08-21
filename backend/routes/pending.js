const express = require('express');
const {pending} = require('../models/PendingModel');
const {addHolding} = require('./holdings');
const { order } = require('../models/OrderModel');
const router = express.Router();
const {finnhubClient} = require('../services/finnhub');

router.post('/pending', async (req,res)=>{
  let newPending = new pending({
    name:req.body.name,
    price:req.body.price
  })
  newPending.save()
})
// Add route to manually trigger pending order processing
router.post('/process-pending', async (req, res) => {
  try {
    await processPendingOrders();
    res.json({ message: 'Pending orders processed successfully' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

const processPendingOrders = async () => {
  try {
    // Get all pending orders
    const pendingOrders = await pending.find({});
    
    if (pendingOrders.length === 0) {
      console.log('No pending orders to process');
      return;
    }

    await Promise.all(
      pendingOrders.map(async (pendingOrder) => {
        return new Promise((resolve, reject) => {
          finnhubClient.quote(pendingOrder.name, async (error, data) => {
            try {
              if (error) {
                console.error(`Error fetching price for ${pendingOrder.name}:`, error);
                reject(error);
                return;
              }

              const currentPrice = data.c; // Current price from Finnhub
              const orderPrice = pendingOrder.price;

              // Check if current price is lower than or equal to order price
              if (currentPrice <= orderPrice) {
                console.log(`Price condition met for ${pendingOrder.name}: Current: ${currentPrice}, Order: ${orderPrice}`);
                
                // Remove from pending collection
                await pending.deleteOne({ _id: pendingOrder._id });
                
                // Update the order to set isPending to false
                await order.updateOne(
                  { 
                    name: pendingOrder.name, 
                    price: pendingOrder.price,
                    isPending: true 
                  },
                  { 
                    isPending: false 
                  }
                );
                const orderDoc = await order.findOne({ name: pendingOrder.name, price: pendingOrder.price });
                addHolding(pendingOrder.name, orderDoc.qty, pendingOrder.price);
                console.log(`Processed pending order for ${pendingOrder.name}`);
              }
              
              resolve();
            } catch (updateError) {
              console.error(`Error processing pending order for ${pendingOrder.name}:`, updateError);
              reject(updateError);
            }
          });
        });
      })
    );

    console.log('Finished processing pending orders');
  } catch (err) {
    console.error('Error in processPendingOrders:', err);
  }
};

module.exports = {
  pending:router,
  processPendingOrders
};
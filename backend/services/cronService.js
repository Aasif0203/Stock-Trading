const cron = require('node-cron');
const {pending} = require('../models/PendingModel');
const {order} = require('../schemas/OrderSchema');

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

const scheduledTask = cron.schedule('*/30 * * * *', async () =>{
    await processPendingOrders();
  }, {
  scheduled: true ,
  timezone: "Asia/Kolkata"
});

// You can start and stop the task programmatically
scheduledTask.start();
// scheduledTask.stop();

console.log('Cron job scheduled to run every half an hour.');
module.exports = processPendingOrders;
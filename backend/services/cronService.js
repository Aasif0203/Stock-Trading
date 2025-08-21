const cron = require('node-cron');
const {pending} = require('../models/PendingModel');
const {order} = require('../schemas/OrderSchema');
const {updateHoldingsWithRealTimeData} = require('../routes/holdings');
const {processPendingOrders} = require('../routes/pending');



const scheduledTask = cron.schedule('* * * * *', async () =>{
    await processPendingOrders();
  }, {
  scheduled: true ,
  timezone: "Asia/Kolkata"
});

// Schedule holdings update every 5 minutes during market hours (9:30 AM to 3:30 PM IST)
const holdingsUpdateTask = cron.schedule('* * * * *', async () => {
  try {
    console.log('Running scheduled holdings update...');
    await updateHoldingsWithRealTimeData();
  } catch (error) {
    console.error('Scheduled holdings update failed:', error);
  }
}, {
  scheduled: true,
  timezone: "Asia/Kolkata"
});

// You can start and stop the tasks programmatically
scheduledTask.start();
holdingsUpdateTask.start();
// scheduledTask.stop();
// holdingsUpdateTask.stop();

module.exports = {
  processPendingOrders,
  updateHoldingsWithRealTimeData
};
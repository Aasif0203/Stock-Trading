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

const holdingsUpdateTask = cron.schedule('* * * * *', async () => {
  await updateHoldingsWithRealTimeData();
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
const express = require('express');
const {pending} = require('../models/PendingModel');
const router = express.Router();

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

module.exports = router;
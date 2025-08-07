const {holding} = require('../models/HoldingModel');
const express = require('express');
const router = express.Router();

router.get('/allHoldings',async (req,res)=>{
  res.json(await holding.find({}));
  
});

module.exports = router;
const {holding} = require('../models/HoldingModel');
const {getStockQuote} = require('../services/finnhub');
const express = require('express');
const router = express.Router();

router.get('/allHoldings',async (req,res)=>{
  res.json(await holding.find({}));
  
});
let addHolding = async (name,qty,price)=>{
  const existingStock = await holding.findOne({name});
  
  if(existingStock) {
    // Update existing holding: calculate new average price and total quantity
    const totalQty = existingStock.qty + qty;
    const totalInvestment = existingStock.invested + (price * qty);
    const loss = totalInvestment>(totalQty*price);
    const updatedHolding = await holding.findOneAndUpdate(
      {name},
      {
        qty: totalQty,
        invested: totalInvestment,
        price: price,
        isLoss : loss
      },
      {new: true, runValidators:true}
    );
    
  } else {
    // Create new holding
    const newHolding = new holding({
      name,
      qty,
      invested: price*qty,
      price: price, 
      net: "0.00%", // Will be calculated based on market price vs avg price
      day: "0.00%", // Daily change
      isLoss: false
    });
    
    await newHolding.save();
  }
}

// Function to update all holdings with real-time data from Finnhub
const updateHoldingsWithRealTimeData = async () => {
  try {
    console.log('Starting holdings update with real-time data...');
    
    // Get all holdings
    const allHoldings = await holding.find({});
    
    if (allHoldings.length === 0) {
      console.log('No holdings found to update');
      return;
    }
    for (const holdingItem of allHoldings) {
      // Get real-time quote from Finnhub
      const quote = await getStockQuote(holdingItem.name);
      
      const currentPrice = quote.c; // Current price
      const previousClose = quote.pc; // Previous close price
      const invested = holdingItem.invested;
      
      const dayChange = previousClose !== 0 ? ((currentPrice - previousClose) / previousClose * 100).toFixed(2)
        : "0.00";
      let netChange = (holdingItem.qty*currentPrice)-invested;
      
      // Determine if it's a loss
      const isLoss = netChange<0;
      
      // Update the holding
      await holding.findByIdAndUpdate(holdingItem._id, {
        price: currentPrice,
        net: `${netChange}%`,
        day: `${dayChange}%`,
        isLoss: isLoss
      });
      
      console.log(`Updated ${holdingItem.name}: Price=${currentPrice}, Net=${netChange}%, Day=${dayChange}%`);
      }
    }
   catch (error) {
      console.error('Error updating holdings with real-time data:', error);
      throw error;
    }
};

// Function to sell holdings
const sellHolding = async (name, qty, sellPrice = null) => {
  try {
    const existingStock = await holding.findOne({name});
    
    if (!existingStock) {
      throw new Error(`You don't own any shares of ${name}`);
    }
    
    if (existingStock.qty < qty) {
      throw new Error(`Insufficient shares. You own ${existingStock.qty} shares but trying to sell ${qty}`);
    }
    
    // Get real-time price if not provided
    let currentPrice = sellPrice;
    if (!currentPrice) {
      console.log(`Getting real-time price for ${name}...`);
      try {
        const quote = await getStockQuote(name);
        currentPrice = quote.c; // Current price from Finnhub
        console.log(`Real-time price for ${name}: $${currentPrice}`);
      } catch (priceError) {
        console.error(`Failed to get real-time price for ${name}:`, priceError.message);
        throw new Error(`Unable to get current price for ${name}. Please provide a price manually.`);
      }
    }
    
    // Validate price
    if (!currentPrice || currentPrice <= 0) {
      throw new Error('Invalid price. Price must be a positive number.');
    }
    
    // Calculate average price from existing data
    const avgPrice = existingStock.invested / existingStock.qty;
    
    if (existingStock.qty === qty) {
      // Selling all shares - remove the holding completely
      await holding.findOneAndDelete({name});
      console.log(`Sold all ${qty} shares of ${name}. Holding removed.`);
    } else {
      // Selling partial shares - update the holding
      const remainingQty = existingStock.qty - qty;
      const soldInvestment = qty * avgPrice; // Calculate proportional investment
      const remainingInvestment = existingStock.invested - soldInvestment;
      
      // Ensure we don't get NaN values
      if (isNaN(remainingInvestment) || isNaN(remainingQty)) {
        throw new Error('Calculation error: Invalid investment or quantity values');
      }
      
      await holding.findOneAndUpdate(
        {name},
        {
          qty: remainingQty,
          invested: remainingInvestment,
          price: currentPrice // Update with current market price
        },
        {new: true, runValidators: true}
      );
      
      console.log(`Sold ${qty} shares of ${name}. ${remainingQty} shares remaining.`);
    }
    
    return {
      message: `Successfully sold ${qty} shares of ${name}`,
      soldQuantity: qty,
      soldPrice: currentPrice,
      totalValue: qty * currentPrice,
      avgPrice: avgPrice
    };
    
  } catch (error) {
    console.error('Error selling holding:', error);
    throw error;
  }
};


module.exports = {
  holding : router,
  addHolding,
  sellHolding,
  updateHoldingsWithRealTimeData
}
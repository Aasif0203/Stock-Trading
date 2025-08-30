const {holding} = require('../models/HoldingModel');
const {getStockQuote} = require('../services/finnhub');
const { requireAuth } = require('../util/AuthMiddleware');
const { getUserFunds, updateUserFunds } = require('../util/FundsController');
const express = require('express');
const router = express.Router();

router.get('/allHoldings', requireAuth, async (req,res)=>{
  // Only get holdings for the current user
  res.json(await holding.find({ userId: req.user._id }));
  
});
let addHolding = async (name, qty, price, userId) => {
  // Calculate total cost
  const totalCost = price * qty;
  
  // Check if user has sufficient funds
  const currentFunds = await getUserFunds(userId);
  if (currentFunds < totalCost) {
    throw new Error(`Insufficient funds. You have $${currentFunds} but need $${totalCost}`);
  }
  
  const existingStock = await holding.findOne({ name, userId });
  
  if(existingStock) {
    // Update existing holding: calculate new average price and total quantity
    const totalQty = existingStock.qty + qty;
    const totalInvestment = existingStock.invested + (price * qty);
    const loss = totalInvestment>(totalQty*price);
    const updatedHolding = await holding.findOneAndUpdate(
      { name, userId },
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
      userId,  // Add userId when creating new holding
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
  
  // Deduct funds from user account
  const newFunds = currentFunds - totalCost;
  await updateUserFunds(userId, newFunds);
  
  return {
    message: `Successfully bought ${qty} shares of ${name}`,
    totalCost: totalCost,
    remainingFunds: newFunds
  };
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
const sellHolding = async (name, qty, sellPrice = null, userId) => {
  try {
    const existingStock = await holding.findOne({ name, userId });
    
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
      await holding.findOneAndDelete({ name, userId });
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
        { name, userId },
        {
          qty: remainingQty,
          invested: remainingInvestment,
          price: currentPrice // Update with current market price
        },
        {new: true, runValidators: true}
      );
      
      console.log(`Sold ${qty} shares of ${name}. ${remainingQty} shares remaining.`);
    }
    
    // Add funds from sale to user account
    const saleValue = qty * currentPrice;
    const currentFunds = await getUserFunds(userId);
    const newFunds = currentFunds + saleValue;
    await updateUserFunds(userId, newFunds);
    
    return {
      message: `Successfully sold ${qty} shares of ${name}`,
      soldQuantity: qty,
      soldPrice: currentPrice,
      totalValue: saleValue,
      avgPrice: avgPrice,
      newFunds: newFunds
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
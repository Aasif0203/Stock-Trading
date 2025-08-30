const express = require('express');
const router = express.Router();
const {watchlist} = require('../models/WatchListModel');
const { requireAuth } = require('../util/AuthMiddleware');
const finnhub = require('finnhub');
const finnhubClient = new finnhub.DefaultApi(process.env.FINNHUB_API_KEY);

router.get('/watchlist', requireAuth, async (req,res)=>{
  // Only get watchlist for current user
  res.json(await watchlist.find({ userId: req.user._id }));
})
// UPDATE ALL DATA in WatchList
router.put('/watchlist', requireAuth, async (req, res) => {
  try {
    // Only update watchlist items for current user
    const symbols = await watchlist.find({ userId: req.user._id }).select('name -_id');
    await Promise.all(
      symbols.map(async ({ name }) => {
        return new Promise((resolve, reject) => {
          finnhubClient.quote(name, async (error, data) => {
            try {
              if (error) {
                reject(error);
              } else {
                await watchlist.updateOne(
                  { name, userId: req.user._id }, // Only update user's own watchlist items
                  {
                    currentPrice: data.c,
                    percentChange: data.d,
                    highPrice: data.h,
                    lowPrice: data.l,
                    openPrice: data.o,
                    previousClose: data.pc,
                    isLoss: data.d < 0,
                  }
                );
                resolve();
              }
            } catch (updateError) {
              reject(updateError);
            }
          });
        });
      })
    );
    res.json({ message: 'Watchlist updated successfully' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/watchlist/:symbol', requireAuth, async (req,res)=>{
  let { symbol } = req.params;
  // Check if symbol already exists for this user
  const existingSymbol = await watchlist.findOne({ name: symbol, userId: req.user._id });
  if (existingSymbol) {
    throw new Error('Stock already exists in your watchlist!');
  }

  const stockData = await new Promise((resolve, reject) => {
    finnhubClient.quote(symbol, (error, data) => {
      if (error) {
        reject(error);
      } else {
        resolve(data);
      }
    });
  });
  if(stockData.d==null) throw new Error('Stock ticker not found !');
  
  let newWatchList = new watchlist({
    userId: req.user._id,  // Add user ID
    name: symbol,
    currentPrice: stockData.c,
    percentChange: stockData.d,
    highPrice: stockData.h,
    lowPrice: stockData.l,
    openPrice: stockData.o,
    previousClose: stockData.pc,
    isLoss: (stockData.d < 0),
  });

  const savedItem = await newWatchList.save();
  res.status(201).json(savedItem); 
});

router.post('/deleteWatchlist/:name', requireAuth, async (req, res) => {
  let { name } = req.params;
  // Only delete from current user's watchlist
  await watchlist.findOneAndDelete({ name, userId: req.user._id });
  console.log("deleted!!");
  res.sendStatus(204);
})
module.exports = router;
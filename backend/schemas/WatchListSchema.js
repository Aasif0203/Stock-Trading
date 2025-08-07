const {Schema} = require('mongoose');

exports.WatchListSchema = new Schema({
  name: String,
  currentPrice: Number,
  percentChange: Number,
  highPrice: Number,
  lowPrice: Number,
  openPrice: Number,
  previousClose: Number,
  isLoss: Boolean
})

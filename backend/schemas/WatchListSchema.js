const {Schema} = require('mongoose');

exports.WatchListSchema = new Schema({
  userId: {
    type: Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  name: String,
  currentPrice: Number,
  percentChange: Number,
  highPrice: Number,
  lowPrice: Number,
  openPrice: Number,
  previousClose: Number,
  isLoss: Boolean
})

const {Schema} = require('mongoose');

exports.PositionSchema = new Schema({
    product: String,
    name: String,
    qty: Number,
    avg: Number,
    price: Number,
    net: String,
    day: String,
    isLoss:{
      type: Boolean,
      default: false
    }
});
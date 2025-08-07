const {Schema} = require('mongoose');

exports.OrderSchema = new Schema({
  name: String,
  qty:Number,
  price: Number,
  mode: String,
  isPending:Boolean,
  time: String,
  day:String
});
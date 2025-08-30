const {Schema} = require('mongoose');

exports.OrderSchema = new Schema({
  userId: {
    type: Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  name: String,
  qty:Number,
  price: Number,
  mode: String,
  isPending:Boolean,
  time: String,
  day:String
});
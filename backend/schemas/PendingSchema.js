const {Schema} = require('mongoose');

exports.PendingSchema = new Schema({
  userId: {
    type: Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  name: String,
  targetPrice: Number,
  qty: Number,
  mode: String
});
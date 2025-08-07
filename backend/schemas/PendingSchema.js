const {Schema} = require('mongoose');

exports.PendingSchema = new Schema({
  name: String,
  price: Number
});
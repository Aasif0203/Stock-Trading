const {model} = require('mongoose');
const {OrderSchema} = require('../schemas/OrderSchema');

exports.order = model('order',OrderSchema);

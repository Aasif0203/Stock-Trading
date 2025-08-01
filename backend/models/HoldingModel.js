const {model} = require('mongoose');
const {HoldingsSchema} = require('../schemas/HoldingsSchema');

const holding = model('holding',HoldingsSchema);
module.exports = {holding};

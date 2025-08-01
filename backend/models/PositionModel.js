const {model}= require('mongoose');
const {PositionSchema} = require('../schemas/PositionSchema');

exports.position = model('position',PositionSchema);
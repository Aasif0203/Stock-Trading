const {model} = require('mongoose');
const {PendingSchema} = require('../schemas/PendingSchema');

exports.pending = model('pending',PendingSchema);
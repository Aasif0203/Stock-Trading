const {model} = require('mongoose');
const {WatchListSchema}= require('../schemas/WatchListSchema');
exports.watchlist = model('watchlist',WatchListSchema);
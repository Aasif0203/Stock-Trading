require('dotenv').config();
const express = require('express');
const app = express();
const mongoose = require('mongoose');
const cors = require('cors');


const PORT = process.env.PORT || 3001;
const uri = process.env.MONGO_URL;

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cors());

const {holding} = require('./routes/holdings');
const ordersRoutes = require('./routes/orders');
const watchlistRoutes = require('./routes/watchlist');
const {pending} = require('./routes/pending');

// Import cronService to start the scheduled tasks
require('./services/cronService');

app.use('/',holding);
app.use('/',ordersRoutes);
app.use('/',watchlistRoutes);
app.use('/',pending);

app.listen(3001,()=>{
  console.log(`app connection successful in PORT ${PORT}`);
  mongoose.connect(uri);
  console.log('DB connected!!');
})
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

const holdingsRoutes = require('./routes/holdings');
const ordersRoutes = require('./routes/orders');
const watchlistRoutes = require('./routes/watchlist');
const pendingRoutes = require('./routes/pending');

app.use('/',holdingsRoutes);
app.use('/',ordersRoutes);
app.use('/',watchlistRoutes);
app.use('/',pendingRoutes);

app.listen(3001,()=>{
  console.log(`app connection successful in PORT ${PORT}`);
  mongoose.connect(uri);
  console.log('DB connected!!');
})
require('dotenv').config();
const express = require('express');
const app = express();
const mongoose = require('mongoose');
const cors = require('cors');
const cookieParser = require("cookie-parser");

const PORT = process.env.PORT || 3001;
const uri = process.env.MONGO_URL;

// Middleware setup (MUST be before routes)
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cors({
  origin: ["http://localhost:5173", "http://localhost:3000"], // Frontend URLs (Vite uses 5173, React dev server uses 3000)
  methods: ["GET", "POST", "PUT", "DELETE"],
  credentials: true // Important for cookies to work
}));
app.use(cookieParser()); 

// Import routes
const authRoute = require("./routes/AuthRoute");
const {holding} = require('./routes/holdings');
const ordersRoutes = require('./routes/orders');
const watchlistRoutes = require('./routes/watchlist');
const {pending} = require('./routes/pending');
const fundsRoutes = require('./routes/funds');

require('./services/cronService');

app.use("/", authRoute);  
app.use("/", holding);     
app.use("/", ordersRoutes);
app.use("/", watchlistRoutes); 
app.use("/", pending);
app.use("/", fundsRoutes);  // Add funds routes     

app.listen(PORT, () => {
  console.log(`App connection successful on PORT ${PORT}`);
  mongoose.connect(uri);
  console.log('DB connected!!');
})
# Stock Trading Platform - Project Report

## Abstract

This project presents a comprehensive **Stock Trading Platform** developed as a full-stack web application using Node.js, Express.js, MongoDB, and real-time financial data integration. The platform simulates a complete stock trading experience, enabling users to manage portfolios, execute trades, track watchlists, and monitor pending orders with real-time market data from Finnhub API. The system incorporates advanced features including automated order execution, secure authentication, funds management, and real-time price updates through scheduled cron jobs.

## Problem Statement

New investors face significant barriers when learning stock trading due to high minimum investments, complex interfaces, and financial risks associated with real trading platforms. Existing educational trading simulators often lack real-time data integration and provide oversimplified experiences that don't reflect actual market conditions. Additionally, individual investors struggle with portfolio management complexities, including tracking multiple holdings, calculating profits/losses, and managing pending orders efficiently. The absence of realistic practice environments with live market data reduces the authenticity of the learning experience, creating a gap between theoretical knowledge and practical trading skills.

## Solution

The solution implements a **RESTful API-based architecture** using Node.js, Express.js, and MongoDB to create a comprehensive stock trading simulator. The platform integrates real-time market data through Finnhub API and implements secure JWT authentication for user management. Core functionalities include portfolio holdings management with automatic profit/loss calculations, advanced order processing supporting both market and limit orders, dynamic watchlist monitoring, and automated background processing via Node-Cron for pending order execution and real-time price updates.

The system features a modular design with separate routes for authentication, holdings, orders, watchlist, and funds management. Security is ensured through JWT token authentication, user data isolation, and bcrypt password encryption. Real-time capabilities include live price updates, automatic order execution when target prices are met, and continuous portfolio valuation updates, providing users with an authentic trading experience while maintaining complete financial safety through simulated transactions.

## Technical Architecture & Database Design

### Database Schemas Implementation

The platform utilizes **MongoDB with Mongoose ODM** for flexible data modeling, implementing five core schemas that handle different aspects of the trading system:

#### 1. **User Schema** - Authentication & Fund Management
```javascript
const userSchema = new mongoose.Schema({
  email: { type: String, required: true, unique: true },
  username: { type: String, required: true },
  password: { type: String, required: true }, // bcrypt hashed
  funds: { type: Number, default: 10000, min: 0 },
  createdAt: { type: Date, default: Date.now }
});
```

#### 2. **Holdings Schema** - Portfolio Management
```javascript
const HoldingsSchema = new Schema({
  userId: { type: Schema.Types.ObjectId, ref: 'User', required: true },
  name: String,           // Stock symbol
  qty: Number,           // Quantity owned
  invested: Number,      // Total amount invested
  price: Number,         // Current/Last price
  net: String,          // Net profit/loss percentage
  day: String,          // Daily change percentage
  isLoss: Boolean       // Quick loss indicator
});
```

#### 3. **Order Schema** - Transaction History
```javascript
const OrderSchema = new Schema({
  userId: { type: Schema.Types.ObjectId, ref: 'User', required: true },
  name: String,       
  qty: Number,          
  price: Number,       
  mode: String,         // 'BUY' or 'SELL'
  isPending: Boolean,   
  time: String,         
  day: String          
});
```

#### 4. **Pending Orders Schema** - Limit Orders Management
```javascript
const PendingSchema = new Schema({
  userId: { type: Schema.Types.ObjectId, ref: 'User', required: true },
  name: String,
  targetPrice: Number,           // Price target for execution
  qty: Number,
  mode: { type: String, enum: ['BUY', 'SELL'] },
  orderType: { type: String, default: 'LIMIT' },
  status: { type: String, default: 'PENDING', 
            enum: ['PENDING', 'EXECUTED', 'CANCELLED'] },
  createdAt: { type: Date, default: Date.now }
});
```

#### 5. **Watchlist Schema** - Stock Monitoring
```javascript
const WatchListSchema = new Schema({
  userId: { type: Schema.Types.ObjectId, ref: 'User', required: true },
  name: String,              // Stock symbol
  currentPrice: Number,      // Real-time price
  percentChange: Number,     // Daily change percentage
  highPrice: Number,         // Daily high
  lowPrice: Number,          // Daily low
  openPrice: Number,         // Opening price
  previousClose: Number,     // Previous close
  isLoss: Boolean           // Loss indicator
});
```

### Key Technical Components

#### **Real-time Data Integration Service**
```javascript
// Finnhub API integration for live market data
const getStockQuote = async (symbol) => {
  const response = await axios.get(`${FINNHUB_BASE_URL}/quote`, {
    params: { symbol: symbol.toUpperCase(), token: FINNHUB_API_KEY }
  });
  return {
    currentPrice: data.c, change: data.d, percentChange: data.dp,
    highPrice: data.h, lowPrice: data.l, openPrice: data.o, previousClose: data.pc
  };
};
```

#### **Automated Cron Service**
```javascript
// Background processing for order execution and price updates
const scheduledTask = cron.schedule('* * * * *', async () => {
  await processPendingOrders();      // Check and execute limit orders
}, { scheduled: true, timezone: "Asia/Kolkata" });

const holdingsUpdateTask = cron.schedule('* * * * *', async () => {
  await updateHoldingsWithRealTimeData();  // Update portfolio values
}, { scheduled: true, timezone: "Asia/Kolkata" });
```

#### **JWT Authentication Middleware**
```javascript
// Secure route protection and user verification
const requireAuth = async (req, res, next) => {
  const token = req.cookies.token;
  jwt.verify(token, process.env.TOKEN_KEY, async (err, data) => {
    if (err) return res.status(401).json({ message: "Invalid token" });
    const user = await User.findById(data.id);
    req.user = user;  // Attach user to request
    next();
  });
};
```

## Technical Implementation Details

### Database Schema Design:
- **Users Collection**: Authentication, profile, and fund management
- **Holdings Collection**: Portfolio positions and investment tracking
- **Orders Collection**: Complete trade history and execution records
- **Watchlist Collection**: User-specific stock monitoring lists
- **Pending Orders Collection**: Limit orders awaiting execution

### API Endpoints Structure:
- Authentication routes (`/signup`, `/login`, verification)
- Holdings management (`/allHoldings`, buy/sell operations)
- Order processing (`/addOrder`, `/allOrders`)
- Watchlist operations (`/watchlist`, add/remove stocks)
- Pending orders (`/pending`, cancellation, status tracking)
- Funds management (`/funds`, `/addFunds`)

### Security Features:
- **JWT Token Authentication**: Secure API access control
- **User Data Isolation**: Database queries filtered by user ID
- **Input Validation**: Request sanitization and error handling
- **Environment Variables**: Sensitive data protection

### Real-time Features:
- **Live Price Updates**: Minute-by-minute market data refresh
- **Automatic Order Execution**: Background processing of limit orders
- **Dynamic P&L Calculation**: Real-time profit/loss computation
- **Market Indicators**: Live change percentages and price movements

## Key Features Implemented

### 1. **Comprehensive Trading Operations**
- Execute buy/sell orders with real-time market prices
- Support for fractional and whole share purchases
- Automatic calculation of transaction costs and remaining funds
- Portfolio diversification tracking across multiple stocks

### 2. **Advanced Order Types**
- **Market Orders**: Immediate execution at current market price
- **Limit Orders**: Conditional execution when price targets are met
- **Order Queuing**: Efficient processing of multiple pending orders
- **Order History**: Complete audit trail of all transactions

### 3. **Real-time Portfolio Analytics**
- Live portfolio valuation based on current market prices
- Individual stock performance tracking with P&L indicators
- Daily change percentages and price movement analysis
- Visual profit/loss indicators for quick assessment

### 4. **Intelligent Watchlist Management**
- Monitor unlimited stocks without investment commitment
- Real-time price tracking for potential investment opportunities
- Quick-trade functionality directly from watchlist
- Market trend analysis for monitored securities

### 5. **Automated System Operations**
- **Background Price Updates**: Continuous market data synchronization
- **Smart Order Execution**: Automatic processing when conditions are met
- **Performance Optimization**: Efficient batch processing of updates
- **Error Handling**: Robust system recovery and error reporting

### 6. **User Experience Enhancements**
- Intuitive API responses with detailed transaction information
- Comprehensive error messaging for failed operations
- Real-time balance updates after each transaction
- Detailed order confirmation and execution reports

## Technical Achievements

### 1. **Scalable Architecture**
- Modular route structure for easy maintenance and expansion
- Separation of concerns between models, controllers, and services
- Reusable utility functions for common operations
- Environment-based configuration management

### 2. **Data Integrity & Validation**
- Mongoose schema validation for data consistency
- Transaction atomicity for critical operations
- Input sanitization and type checking
- Comprehensive error handling and user feedback

### 3. **Performance Optimization**
- Efficient database queries with proper indexing
- Batch processing for bulk operations
- Caching strategies for frequently accessed data
- Optimized API response structures

### 4. **Security Implementation**
- JWT token-based authentication system
- Secure password hashing with bcrypt
- User session management and timeout handling
- API endpoint protection and authorization

## Summary

I completed my 9-week Summer Research Internship at the Department of Computer Science and Engineering (CSE), working under the guidance of Associate Professor Dr. A. Santhanavijayan. The duration of the internship was from 16th May 2025 to 18th July 2025.

During the first four weeks, I focused on designing the frontend components of the Stock Trading Platform, which involved creating an intuitive dashboard UI, interactive watchlist sections, and user authentication interfaces. The next four weeks were dedicated to developing the robust backend architecture using Node.js and Express.js, implementing core functionalities including user management, portfolio tracking, order processing, and real-time data integration.

The final week was dedicated to advanced system integration, comprehensive user authentication implementation, and the design of an LSTM (Long Short-Term Memory) neural network model for intelligent stock price prediction. This phase involved integrating machine learning capabilities to analyze historical market data patterns and provide predictive insights for enhanced trading decisions. The project culminated in a comprehensive stock trading simulator that not only bridges the gap between theoretical financial knowledge and practical trading experience but also incorporates machine learning techniques for predictive market analysis.

### Key Technical Accomplishments:

- **Full-Stack Development**: Built complete trading platform with React frontend and Node.js backend
- **Database Design**: Implemented MongoDB schemas for users, holdings, orders, watchlist, and pending transactions
- **Real-time Integration**: Connected live market data through Finnhub API with automated price updates
- **Security Implementation**: Developed secure authentication system with JWT tokens and bcrypt encryption
- **Automated Processing**: Created background services for limit order execution and portfolio value updates
- **API Architecture**: Designed RESTful APIs for seamless frontend-backend communication

The internship provided invaluable experience in modern web development technologies, financial system design, and project management, resulting in a comprehensive trading platform that demonstrates both technical proficiency and practical application of computer science principles in the financial domain.

## Future Potential & Enhancements

The Stock Trading Platform provides a solid foundation for advanced features and improvements. The modular architecture supports expansion into sophisticated trading functionalities and cutting-edge financial technology implementations.

### **Key Enhancement Areas:**

- **Advanced Order Types**: Implementation of Stop-Loss orders, Trailing Stop orders, and Bracket orders for comprehensive risk management
- **Enhanced LSTM Model**: Extending prediction capabilities to forecast stock prices for longer timeframes up to one month with improved accuracy
- **Interactive Forecast Visualization**: Developing sophisticated charting interfaces with prediction graphs and confidence intervals
- **Mobile-Responsive Design**: Complete frontend redesign for seamless experience across desktop, tablet, and mobile devices
- **Real-time Analytics**: Advanced portfolio analytics dashboard with risk assessment and performance benchmarking tools
- **API Integration**: Third-party market data sources and additional trading tools for enhanced functionality

---

**Project Timeline**: Summer 2025  
**Technology Stack**: Node.js, Express.js, MongoDB, Mongoose, JWT, Finnhub API, Node-Cron  
**Repository**: [stock-trading](https://github.com/Aasif0203/stock-trading)  
**Status**: Completed ✅

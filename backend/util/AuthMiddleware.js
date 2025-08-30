const User = require("../models/UserModel");
require("dotenv").config();
const jwt = require("jsonwebtoken");

// Helper function to verify token and get user (removes code duplication)
const verifyTokenAndGetUser = async (token) => {
  return new Promise((resolve, reject) => {
    if (!token) {
      return reject({ status: false, message: "No token provided" });
    }
    
    jwt.verify(token, process.env.TOKEN_KEY, async (err, data) => { 
      if (err) {
        return reject({ status: false, message: "Invalid token" });
      }
      
      try {
        const user = await User.findById(data.id);
        if (user) {
          resolve({ status: true, user });
        } else {
          reject({ status: false, message: "User not found" });
        }
      } catch (error) {
        reject({ status: false, message: "Database error" });
      }
    });
  });
};

// Route endpoint: Check if user is logged in (returns status to frontend)
module.exports.userVerification = async (req, res) => {
  try {
    const token = req.cookies.token;
    const result = await verifyTokenAndGetUser(token);
    res.json({ status: true, user: result.user.username });
  } catch (error) {
    res.json({ status: false });
  }
};

// Middleware: Protect routes from unauthorized access
module.exports.requireAuth = async (req, res, next) => {
  try {
    const token = req.cookies.token;
    const result = await verifyTokenAndGetUser(token);
    req.user = result.user; // Add user info to request object
    next(); // Continue to the protected route
  } catch (error) {
    res.status(401).json(error);
  }
};

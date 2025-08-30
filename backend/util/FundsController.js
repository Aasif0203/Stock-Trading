const User = require("../models/UserModel");

// Get user's current funds
const getUserFunds = async (userId) => {
  const user = await User.findById(userId);
  return user ? user.funds : 0;
};

// Update user's funds (can add or subtract)
const updateUserFunds = async (userId, newAmount) => {
  const user = await User.findByIdAndUpdate(
    userId, 
    { funds: newAmount }, 
    { new: true }
  );
  return user.funds;
};

module.exports = {
  getUserFunds,
  updateUserFunds
};

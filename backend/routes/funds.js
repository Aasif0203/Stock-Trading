const express = require('express');
const router = express.Router();
const { requireAuth } = require('../util/AuthMiddleware');
const { getUserFunds } = require('../util/FundsController');
const User = require('../models/UserModel');

// Get current user's fund balance - that's it!
router.get('/funds', requireAuth, async (req, res) => {
  const funds = await getUserFunds(req.user._id);
  res.json({ funds: funds });
});

// Add funds to user account
router.post('/addFunds', requireAuth, async (req, res) => {
  const { amount } = req.body;
  const user = await User.findById(req.user._id);
  user.funds += Number(amount);
  await user.save();
  res.json({ funds: user.funds });
});

module.exports = router;

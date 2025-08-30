const axios = require('axios');

const FINNHUB_API_KEY = process.env.FINNHUB_API_KEY;
const FINNHUB_BASE_URL = 'https://finnhub.io/api/v1';

const getStockQuote = async (symbol) => {
  if (!FINNHUB_API_KEY) {
    throw new Error('FINNHUB_API_KEY is not set in environment variables');
  }

  const response = await axios.get(`${FINNHUB_BASE_URL}/quote`, {
    params: {
      symbol: symbol.toUpperCase(),
      token: FINNHUB_API_KEY
    }
  });

  // Return raw data in the same format your existing code expects
  return response.data;
};

const getCompanyProfile = async (symbol) => {
  if (!FINNHUB_API_KEY) {
    throw new Error('FINNHUB_API_KEY is not set in environment variables');
  }

  const response = await axios.get(`${FINNHUB_BASE_URL}/stock/profile2`, {
    params: {
      symbol: symbol.toUpperCase(),
      token: FINNHUB_API_KEY
    }
  });

  return response.data;
};

module.exports = {
  getStockQuote,
  getCompanyProfile
};
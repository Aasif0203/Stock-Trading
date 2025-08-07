const finnhub = require('finnhub');

const finnhubClient = new finnhub.DefaultApi(process.env.FINNHUB_API_KEY);

const getStockQuote = (symbol) => {
  return new Promise((resolve, reject) => {
    finnhubClient.quote(symbol, (error, data) => {
      if (error) {
        reject(error);
      } else {
        resolve(data);
      }
    });
  });
};

module.exports = {
  finnhubClient,
  getStockQuote
};
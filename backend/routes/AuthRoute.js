const { Signup,Login } = require("../util/AuthController");
const { userVerification } = require("../util/AuthMiddleware");
const router = require("express").Router();

router.post("/signup", Signup);
router.post("/login",Login);
router.post('/',userVerification);

module.exports = router;
import React from "react";
import { Link } from "react-router-dom";
import ButtonGroup from '@mui/material/ButtonGroup';
import Button from '@mui/material/Button';

const Funds = () => {
  return (
    <>
      <div className="funds" style={{ color: "white" }}>
        <p>Instant, zero-cost fund transfers with UPI </p> &nbsp;
        
        <ButtonGroup variant="contained" >
          <Button color='info'>
            <Link style={{ color: "white", textDecoration: "none" }}>Add funds</Link>
          </Button>
          <Button>
            <Link style={{ color: "white", textDecoration: "none" }}>Withdraw</Link>
          </Button>
        </ButtonGroup>
      </div>

      <div style={{ color: "white", display:'flex' }} >
        <div >
          <span>
            <h4 style={{textAlign:'start', padding:'0px 20px'}} >Equity</h4>
          </span>

          <span className="table">
            <div className="data">
              <p>Available margin : </p> &nbsp;
              <p className="imp colored">4,043.10</p>
            </div>
            <div className="data">
              <p>Used margin : </p> &nbsp;
              <p className="imp">3,757.30</p>
            </div>
            <div className="data">
              <p>Available cash</p> &nbsp;
              <p className="imp">4,043.10</p>
            </div>
            <hr className="tabledivider"/>
            <div className="data">
              <p>Opening Balance</p> &nbsp;
              <p>4,043.10</p>
            </div>
            <div className="data"> 
              <p>Opening Balance : </p> &nbsp;
              <p>3736.40</p>
            </div>
            <div className="data">
              <p>Payin : </p> &nbsp;
              <p>4064.00</p>
            </div>
            <div className="data">
              <p>SPAN : </p> &nbsp;
              <p>0.00</p>
            </div>
            <div className="data">
              <p>Delivery margin : </p> &nbsp;
              <p>0.00</p>
            </div>
            <div className="data">
              <p>Exposure : </p> &nbsp;
              <p>0.00</p>
            </div>
            <div className="data">
              <p>Options premium : </p> &nbsp;
              <p>0.00</p>
            </div>
            <hr className="tabledivider"/>
            <div className="data">
              <p>Collateral (Liquid funds) : </p> &nbsp;
              <p>0.00</p>
            </div>
            <div className="data">
              <p>Collateral (Equity) :</p> &nbsp;
              <p>0.00</p>
            </div>
            <div className="data">
              <p>Total Collateral : </p> &nbsp;
              <p>0.00</p>
            </div>
          </span>
        </div>

        <div style={{alignSelf:'center', margin:'auto'}}>
          <div>
            <p>You don't have a commodity account</p>
            <Button variant="contained" color="success"><Link style={{color:'white'}}>Open Account</Link></Button>
          </div>
        </div>
      </div>
    </>
  );
};

export default Funds;

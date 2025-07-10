import * as React from 'react';

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCircleUser } from '@fortawesome/free-solid-svg-icons';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import Menu from '@mui/material/Menu';
import MenuIcon from '@mui/icons-material/Menu';
import Container from '@mui/material/Container';
import Avatar from '@mui/material/Avatar';
import Button from '@mui/material/Button';
import Tooltip from '@mui/material/Tooltip';
import MenuItem from '@mui/material/MenuItem';
import AdbIcon from '@mui/icons-material/Adb';
import { Link } from "react-router-dom";

const pages = ['Orders', 'Holdings','Positions','Funds'];
const settings = ['Profile', 'Account', 'Dashboard', 'Logout'];

function ResponsiveAppBar() {
  const [anchorElNav, setAnchorElNav] = React.useState(0);
  const [anchorElUser, setAnchorElUser] = React.useState(false);

  const handleOpenNavMenu = (key) => {
    setAnchorElNav(key);
  };
  const handleOpenUserMenu = () => {
    setAnchorElUser(!anchorElUser);
  };

  return (
    <AppBar position="static"  color='success' sx={{borderRadius:'9px'}}>
      <Container maxWidth="xl">
        <Toolbar disableGutters>
          
          <Link key="0" to="/" onClick= {() =>{handleOpenNavMenu(0)}} >
            <img src='/logo_sticker.png' alt='logo' id='logo' />
          </Link>

          <Link key="0" to="/" onClick= {() =>{handleOpenNavMenu(0)}} >
            <p className = {anchorElNav===0? 'activeMenu':'justMenu'}>Home</p>
          </Link>
          <Link key="1" to="/Orders" onClick= {() =>{handleOpenNavMenu(1)}} >
            <p className = {anchorElNav===1? 'activeMenu':'justMenu'}>Orders</p>
          </Link>
          <Link key="2" to="/Positions" onClick= {() =>{handleOpenNavMenu(2)}} >
            <p className = {anchorElNav===2? 'activeMenu':'justMenu'}>Positions</p>
          </Link>
          <Link key="3" to="/Holdings" onClick= {() =>{handleOpenNavMenu(3)}} >
            <p className = {anchorElNav===3? 'activeMenu':'justMenu'}>Holdings</p>
          </Link>
          <Link key="4" to="/Funds" onClick= {() =>{handleOpenNavMenu(4)}} >
            <p className = {anchorElNav===4? 'activeMenu':'justMenu'}>Funds</p>
          </Link>

          <Box sx={{ flexGrow: 1, display: { xs: 'none', md: 'flex' }, width: '50%' }} >
            
          </Box>
          {/* <FontAwesomeIcon icon="fa-solid fa-circle-user" width={100} height={100}/> */}
          <Box sx={{ flexGrow: 0 }} width={100}>
            <Tooltip title="Open settings">
              <IconButton onClick={handleOpenUserMenu} sx={{ p: 0 }}>
                <FontAwesomeIcon icon={faCircleUser} style={{ fontSize: '50px', color: 'white' }}/>
              </IconButton>
            </Tooltip>
            <Menu
              sx={{ mt: '45px' }}
              id="menu-appbar"
              anchorEl={anchorElUser}
              anchorOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
              keepMounted
              transformOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
              open={anchorElUser}
            >
              {settings.map((setting) => (
                <MenuItem key={setting} onClick={handleOpenUserMenu}>
                  <Typography sx={{ textAlign: 'center' }}>{setting}</Typography>
                </MenuItem>
              ))}
            </Menu>
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
}
export default ResponsiveAppBar;

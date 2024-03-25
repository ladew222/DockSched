import React from 'react';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import { Box } from '@mui/material';

const Navbar = () => {
    return (
        <Box sx={{ flexGrow: 1 }}>
            <AppBar position="static">
                <Toolbar>
                    <Typography variant="h6" sx={{ flexGrow: 1 }}>
                        Class Scheduler
                    </Typography>
                    <Button color="inherit">Save Schedule</Button>
                    <Button color="inherit">Load Schedule</Button>
                    <Button color="inherit">Help</Button>
                </Toolbar>
            </AppBar>
        </Box>
    );
};

export default Navbar;

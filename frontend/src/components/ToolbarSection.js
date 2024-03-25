import React from 'react';
import { AppBar, Toolbar, Button, Typography, Slider, Box } from '@mui/material';

function ToolbarSection() {
  // Event handlers can be added for file input, sliders, and button clicks
  return (
    <AppBar position="static" color="default" className="appBar">
      <Toolbar className="toolBar">
        <Box className="fileInputContainer">
          <input
            accept=".csv"
            className="fileInputHidden"
            id="contained-button-file"
            multiple
            type="file"
          />
          <label htmlFor="contained-button-file">
            <Button variant="contained" component="span">
              Choose File
            </Button>
          </label>
          <Typography variant="body1" color="textSecondary">
           {fileData.name}
          </Typography>
        </Box>
        <Box className="penaltyContainer">
          <Typography variant="h6" gutterBottom>
            Penalty Parameters
          </Typography>
          <Slider defaultValue={30} />
          {/* Additional sliders can be added here */}
        </Box>
        <Box className="buttonContainer">
          <Button variant="contained" className="button">
            Apply Constraints & Optimize
          </Button>
          <Button variant="contained" color="secondary">
            Hold All
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
}

export default ToolbarSection;

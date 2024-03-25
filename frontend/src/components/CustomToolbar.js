import React, { useState } from 'react';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Slider from '@mui/material/Slider';
import Box from '@mui/material/Box';
import { styled } from '@mui/material/styles';

// Styled components for better control over slider color
const StyledSlider = styled(Slider)(({ theme }) => ({
  color: theme.palette.common.white, // Change thumb color to white
  '& .MuiSlider-track': {
    color: theme.palette.common.white, // Change track color to white
  },
  '& .MuiSlider-rail': {
    color: theme.palette.grey[400], // Change rail color to grey
  },
}));

// Styled Button with contrast and gradient
const GradientButton = styled(Button)(({ theme }) => ({
  background: 'linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)',
  border: '1px solid white',
  borderRadius: 3,
  color: 'white',
  fontWeight: 'bold',
  padding: '10px 30px',
  margin: theme.spacing(1),
  '&:hover': {
    background: 'linear-gradient(45deg, #FF8E53 30%, #FE6B8B 90%)',
  },
}));

const CustomToolbar = ({ fileData, onOptimize, onHoldAll, sliderValues, onSliderChange }) => {
  const [file, setStFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');

  const handleFileChange = (event) => {
    setStFile(event.target.files[0]);
  };

  const handleSf_Upload = () => {
    if (file) {
      const formData = new FormData();
      formData.append('file', file);

      fetch('http://127.0.0.1:5000/upload_student_schedule', {
        method: 'POST',
        body: formData,
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then(result => {
        if (result.error) {
          setUploadStatus(result.error);
          return;
        }
        setUploadStatus('File uploaded successfully.');
        // Optionally, perform an action on successful upload
        // onOptimize(); // You can call onOptimize or similar function if needed
      })
      .catch(error => {
        setUploadStatus('Failed to upload file. ' + error.message);
      });
    } else {
      setUploadStatus('Please select a file first.');
    }
  };

  
  
  return (
    <AppBar position="static" sx={{ backgroundColor: 'navy' }}>
      <Toolbar>
        {/* Logo aligned to the far left and sized small */}
        <img src="/logo.webp" alt="Class Scheduler Logo" style={{ marginRight: 20, height: '40px', flexGrow: 0 }} />
        
        {/* Title next to the logo */}
        {!fileData && (
          <Typography variant="h6" component="div" sx={{ color: 'white' }}>
          Class Scheduler
        </Typography>
        )}
        {fileData && (
        <Box>
          <input
              accept="csv/*"
              style={{ display: 'none' }}
              id="raised-button-St_file"
              type="file"
              onChange={handleFileChange}
          />
          <label htmlFor="raised-button-St_file">
              <Button  size='small' variant="contained" component="span" color="primary"  sx={{ padding: '6px 16px', fontSize: '0.875rem', backgroundColor: 'navy' }}>
                  Student File
              </Button>
          </label>
          <Button
              variant="contained"
              color="primary"
              size='small'
              onClick={handleSf_Upload}
              sx={{ padding: '6px 16px', fontSize: '0.875rem', backgroundColor: 'navy' }}
              disabled={!file}
          >
              Upload
          </Button>
            </Box>  
        )}

        {fileData && (
          <>
            <GradientButton onClick={onOptimize}>Optimize</GradientButton>
            <Typography color="inherit" sx={{ marginLeft: '20px', marginRight: '20px' }}>
              {fileData.name}
            </Typography>
            <Typography color="inherit">
              File size: {(fileData.size / 1024).toFixed(2)} KB
            </Typography>
          </>
        )}

        {/* Sliders displayed if file data is available */}
        {fileData && (
          <Box sx={{ width: '100%', display: 'flex', justifyContent: 'space-around', padding: '10px 0' }}>
            <StyledSliderBox sliderValues={sliderValues} onSliderChange={onSliderChange} />
          </Box>
        )}
      </Toolbar>
    </AppBar>
  );
};

// Component for sliders to keep CustomToolbar clean
const StyledSliderBox = ({ sliderValues, onSliderChange }) => (
  <>
    <Box sx={{ width: '20%', color: 'white' }}>
      <Typography gutterBottom>Avoiding Selected Classes</Typography>
      <StyledSlider
        value={sliderValues.avoidingClassesValue}
        onChange={onSliderChange('avoidingClassesValue')}
        aria-labelledby="Avoiding-Classes-Slider"
      />
    </Box>
    <Box sx={{ width: '20%', color: 'white' }}>
      <Typography gutterBottom>Keep 4 Credit Classes Together</Typography>
      <StyledSlider
        value={sliderValues.keepClassesTogetherValue}
        onChange={onSliderChange('keepClassesTogetherValue')}
        aria-labelledby="Keep-Classes-Together-Slider"
      />
    </Box>
    <Box sx={{ width: '20%', color: 'white' }}>
      <Typography gutterBottom>Blocked Slot Penalty Value</Typography>
      <StyledSlider
        value={sliderValues.blockedSlotPenaltyValue}
        onChange={onSliderChange('blockedSlotPenaltyValue')}
        aria-labelledby="Blocked-Slot-Penalty-Slider"
      />
    </Box>
    <Box sx={{ width: '20%', color: 'white' }}>
      <Typography gutterBottom>Hold Penalty Slider Value</Typography>
      <StyledSlider
        value={sliderValues.holdPenaltyValue}
        onChange={onSliderChange('holdPenaltyValue')}
        aria-labelledby="Hold-Penalty-Slider"
      />
    </Box>
  </>
);

export default CustomToolbar;

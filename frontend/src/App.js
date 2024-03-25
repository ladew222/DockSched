import React, { useState } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import Results from './components/Results';
import FileUpload from './components/FileUpload';
import { Container } from '@mui/material';
import CustomToolbar from './components/CustomToolbar'; // Import your CustomToolbar component



function App() {
  // State for file data and optimization parameters
  const [fileData, setFileData] = useState(null);
  const [fileInfo, setFileInfo] = useState(null);
  const [optimizationTrigger, setOptimizationTrigger] = useState(false);
  const [optimizationParams, setOptimizationParams] = useState({
    avoidingClassesValue: 30,
    keepClassesTogetherValue: 30,
    blockedSlotPenaltyValue: 30,
    holdPenaltyValue: 30,
  });
  const [optimizationResults, setOptimizationResults] = useState(null);
  const [holdAll, setHoldAll] = useState(false);
  const [TableData, setTableData] = useState(false);


  // Handler functions
  const handleFileUploadSuccess = (dataWithFileInfo) => {
    setFileData(dataWithFileInfo.cleanedData);
    setFileInfo(dataWithFileInfo.fileInfo);
  };

  const handleSliderChange = (name) => (event, newValue) => {
    setOptimizationParams((prevParams) => ({
      ...prevParams,
      [name]: newValue,
    }));
  };

  const handleTableDataUpdate = (newTableData) => {
    setTableData(newTableData);
  };
  

  const handleHoldAllToggle = () => {
    setHoldAll(!holdAll);
  };

// In App.js

const handleOptimization = async () => {
  const preparedClassData = fileData.map((row, index) => {
    // Ensure we handle potentially undefined states with default values
    const holdState = TableData.holdCheckedState?.[index] ? '1' : '0';
    const blockedTimeSlots = Array.isArray(TableData.selectedBlockedTimeSlots?.[index]) 
      ? TableData.selectedBlockedTimeSlots[index].join(';') 
      : '';
    const restrictions = Array.isArray(TableData.autocompleteRestrictions?.[index]) 
      ? TableData.autocompleteRestrictions[index].join(';') 
      : '';

    return {
      section: row['sec name'],
      title: row['title'],
      minCredit: row['min Credit'],
      secCap: row['sec Cap'],
      room: row['room'],
      bldg: row['bldg'],
      weekDays: row['week Days'],
      csmStart: row['CSM start'],
      csmEnd: row['CSM end'],
      faculty1: row['faculty1'],
      hold: holdState,
      blockedTimeSlots: blockedTimeSlots,
      restrictions: restrictions,
    };
  });

  const payload = {
    classData: preparedClassData,
    ...optimizationParams, // Assuming optimizationParams is defined elsewhere in your component and contains additional parameters for the optimization request
  };

  try {
    const response = await fetch('http://127.0.0.1:5001/optimize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      // throw new Error('Network response was not ok');
      console.error('Data Processswing Error:', response.status, response.statusText);
      toast.error('Data Processswing Error:'+ response.status + ": " + response.statusText); // Display error notification
    }
    else{
      const result = await response.json();
      setOptimizationResults(result); // Assuming setOptimizationResults updates the state with the optimization results
      toast.success("Optimization successful!"); // Display success notification
    }

   
    
  } catch (error) {
    console.error('Failed to submit data:', error);
    toast.error("Failed to submit data:" + error); // Display error notification
  }
};



  return (
    <div className="App">
        <CustomToolbar
          fileData={fileInfo}
          onOptimize={handleOptimization} // Corrected to match the defined function
          onHoldAll={handleHoldAllToggle}
          sliderValues={optimizationParams}
          onSliderChange={handleSliderChange}
        />
      <Container maxWidth="lg">
        <FileUpload onSuccess={handleFileUploadSuccess} />
        {fileData &&  (
            <Results
            data={fileData}
            onUpdateTableData={handleTableDataUpdate}
            optimizationResults={optimizationResults}
            />
        )}
      </Container>
      <ToastContainer />
    </div>
  );
}

export default App;



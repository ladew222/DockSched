import React, { useState, useEffect } from 'react';
import { Container, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TextField, Accordion, AccordionSummary, AccordionDetails, Typography } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ResultsTable from './ResultsTable';
import Autocomplete from '@mui/material/Autocomplete';
import { Checkbox } from '@mui/material';
import { toast } from 'react-toastify';



const Results = ({ ...props }) => {
  let data=props.data;
  const [holdCheckedState, setHoldCheckedState] = useState(Array(data.length).fill(false));
  //const [selectedBlockedTimeSlots, setSelectedBlockedTimeSlots] = useState(Array(data.length).fill([]));
 // This should be an array of arrays, each initialized to an empty array
  const [selectedBlockedTimeSlots, setSelectedBlockedTimeSlots] = useState(
    data.map(() => [])
  );

  const [isAccordionExpanded, setIsAccordionExpanded] = useState(true);

  const [autocompleteRestrictions, setAutocompleteRestrictions] = useState(Array(data.length).fill([]));
  const [blockedTimeSlotOptions, setBlockedTimeSlotOptions] = useState([]);


   // Close the accordion when optimizationResults are present
   useEffect(() => {
    if (props.optimizationResults) {
      setIsAccordionExpanded(false);
    }
  }, [props.optimizationResults]); // Dependency on optimizationResults

  // useEffect for fetching meeting times - runs only once after component mounts
  useEffect(() => {
    const fetchMeetingTimes = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5001/meeting_times');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const meetingTimes = await response.json();
        // Transform each meeting time object into a string format
        const formattedMeetingTimes = meetingTimes.map(time =>
          `${time.days} - ${time.start_time} to ${time.end_time}`
        );
        setBlockedTimeSlotOptions(formattedMeetingTimes);
      } catch (error) {
        console.error('Error fetching meeting times:', error);
        toast.error(`Error fetching meeting times: ${error.message}`);
        // Handle the error appropriately
      }
    };
    fetchMeetingTimes();
  
  }, []); // Empty dependency array ensures this runs only once


  useEffect(() => {
     // Send only dynamic elements
     if (typeof props.onUpdateTableData === 'function') {
      const dynamicData = {
        holdCheckedState,
        selectedBlockedTimeSlots: selectedBlockedTimeSlots.map(slots => slots.join(';')), // Convert array of slots to string if needed
        autocompleteRestrictions: autocompleteRestrictions.map(restrictions => restrictions.join(';')), // Convert array of restrictions to string if needed
      };
      props.onUpdateTableData(dynamicData);
    }

  }, [holdCheckedState, selectedBlockedTimeSlots, autocompleteRestrictions]); // Only re-run when these states change
  

  useEffect(() => {
    console.log('Results component received new props:', { data });
  }, [data]);
  



  

  const renderTagInput = (value = [], idSuffix) => {
    // Filter out any empty strings to avoid rendering empty tags
    const validValues = value.filter(v => v.trim() !== '');
  
    return (
      <Autocomplete
        multiple
        id={`tags-outlined-${idSuffix}`}
        options={blockedTimeSlotOptions}
        getOptionLabel={(option) => option}
        defaultValue={validValues}
        filterSelectedOptions
        renderInput={(params) => (
          <TextField
            {...params}
            variant="outlined"
            placeholder="Select Blocked Time Slots"
          />
        )}
      />
    );
  };
  
  const handleAutocompleteChange = (index, newValue) => {
    // Update the restrictions for the current row
    const updatedAutocompleteRestrictions = [...autocompleteRestrictions];
    updatedAutocompleteRestrictions[index] = newValue;
    setAutocompleteRestrictions(updatedAutocompleteRestrictions);
  
    // Now, find the row that corresponds to the selected section and add a reciprocal restriction
    newValue.forEach(selectedSection => {
      const targetRowIndex = data.findIndex(row => row['sec name'] === selectedSection);
      if (targetRowIndex !== -1 && targetRowIndex !== index) {
        // Add the current section to the restrictions of the target row, if not already present
        const targetRowRestrictions = new Set([...updatedAutocompleteRestrictions[targetRowIndex], data[index]['sec name']]);
        updatedAutocompleteRestrictions[targetRowIndex] = Array.from(targetRowRestrictions);
      }
    });
  
    // Finally, update the state with the new restrictions
    setAutocompleteRestrictions(updatedAutocompleteRestrictions);
  };
  



  return (
    <Container maxWidth="xl" sx={{ mt: 4 }}>
  
       <Accordion expanded={isAccordionExpanded} onChange={(event, isExpanded) => setIsAccordionExpanded(isExpanded)}>
        <AccordionSummary
          expandIcon={<ExpandMoreIcon />}
          aria-controls="panel1a-content"
          id="panel1a-header"
        >
          <Typography><h1>Loaded Data</h1></Typography>
        </AccordionSummary>
        <AccordionDetails>
      <TableContainer component={Paper}>
        <Table aria-label="schedule table">
          <TableHead>
            <TableRow>
              <TableCell>Section</TableCell>
              <TableCell>Title</TableCell>
              <TableCell>Min Credit</TableCell>
              <TableCell>Sec Cap</TableCell>
              <TableCell>Room</TableCell>
              <TableCell>Bldg</TableCell>
              <TableCell>Week Days</TableCell>
              <TableCell>CSM Start</TableCell>
              <TableCell>CSM End</TableCell>
              <TableCell>Faculty</TableCell>
              <TableCell>Location</TableCell>
              <TableCell>Hold</TableCell>
              <TableCell>Blocked Time Slots</TableCell>
              <TableCell>Restrictions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {data.map((row, index) => (
              <TableRow key={index}>
                <TableCell>{row['sec name']}</TableCell>
                <TableCell>{row.title}</TableCell>
                <TableCell>{row['min Credit']}</TableCell>
                <TableCell>{row['sec Cap']}</TableCell>
                <TableCell>{row.room}</TableCell>
                <TableCell>{row.bldg}</TableCell>
                <TableCell>{row['week Days']}</TableCell>
                <TableCell>{row['CSM start']}</TableCell>
                <TableCell>{row['CSM end']}</TableCell>
                <TableCell>{row.faculty1}</TableCell>
                <TableCell>{`${row.bldg} ${row.room}`}</TableCell>
                <TableCell>
                  <Checkbox
                    checked={holdCheckedState[index]}
                    onChange={() => {
                      const newHoldCheckedState = [...holdCheckedState];
                      newHoldCheckedState[index] = !newHoldCheckedState[index];
                      setHoldCheckedState(newHoldCheckedState);
                    }}
                    inputProps={{ 'aria-label': 'controlled' }}
                  />
                </TableCell>
                <TableCell>{renderTagInput(row['Blocked Time Slots'] ? row['Blocked Time Slots'].split(',') : [])}</TableCell>
 
                <TableCell>
                  <Autocomplete
                    multiple
                    options={data.map((item) => item['sec name'])}
                    getOptionLabel={(option) => option}
                    value={autocompleteRestrictions[index]}
                    onChange={(event, newValue) => handleAutocompleteChange(index, newValue)}
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        variant="outlined"
                        label="Restrictions"
                        placeholder="Select restrictions"
                      />
                    )}
                  />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      </AccordionDetails>
      </Accordion>
      {props.optimizationResults && (
        // Display optimization results. Adjust according to how you want to display them.
        <ResultsTable results={props.optimizationResults} />
      )}
    </Container>
  );
};

export default Results;

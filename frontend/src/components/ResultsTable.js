import React, { useState } from 'react';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Dialog, DialogTitle, DialogContent, DialogActions, Button } from '@mui/material';
import { Calendar, momentLocalizer } from 'react-big-calendar';
import moment from 'moment';
import 'react-big-calendar/lib/css/react-big-calendar.css';
import '../index.css';
import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import ResultsCalendar from './ResultsCalendar';


const ResultsTable = ({ results }) => {

  // State to track the current index of the result being displayed
  const [currentIndex, setCurrentIndex] = useState(0);
  // Computed value to get the current result based on the selected index
  const currentResult = results[currentIndex];

  // Event handler for changing the dropdown selection
  const handleSelectChange = (event) => {
    setCurrentIndex(event.target.value);
  };

  // Function to convert data to CSV format and trigger download
  const handleDownloadCSV = () => {

    try {
      // Define the header for CSV
      const header = ['Section', 'Title', 'Min Credit', 'Timeslot', 'Location', 'Faculty'];
      // Map the data to a CSV format
      const csvContent = [
        header.join(','), // header row first
        ...results[currentIndex].schedule.map(row => [
          row.section,
          row.title,
          row.minCredit,
          row.timeslot,
          `"${row.bldg} ${row.room}"`, // Wrap in quotes to handle any commas
          row.faculty1
        ].join(',')) // join each row's columns with a comma
      ].join('\r\n'); // join each row with a new line

      // Create a Blob with the CSV content and the appropriate type
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      
      // Create a temporary link element and trigger the download
      const link = document.createElement('a');
      link.href = url;
      link.download = 'results.csv'; // File name for download
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
        toast.success("CSV file downloaded successfully!");
    } catch (error) {
        console.error('Error generating CSV:', error);
        toast.error("Failed to download CSV file.");
    }

  };


  return (
    <>
      <div id={'results'}>
        <h1>Optimization Results</h1>
  
        {/* Dropdown to select result */}
        <select onChange={handleSelectChange} value={currentIndex}>
          {results.map((result, index) => (
            <option key={index} value={index}>
              Result {index + 1} - Score: {result.score}
            </option>
          ))}
        </select>
        {/* Display student_conflicts next to the dropdown */}
        {currentResult.student_conflicts && (
            <div>
              <strong>Student Conflicts:</strong> {currentResult.student_conflicts}
            </div>
          )}
        {/* Display current result */}
        <div>
          <h2>Result {currentIndex + 1} - Score: {currentResult.score}</h2>
          <Button onClick={() => handleDownloadCSV(currentResult)}>
            Download CSV for Result {currentIndex + 1}
          </Button>

          <TableContainer component={Paper} style={{ marginBottom: '20px' }}>
              <Table aria-label="optimization results table">
                <TableHead>
                  <TableRow>
                    <TableCell>Section</TableCell>
                    <TableCell>Title</TableCell>
                    <TableCell>Min Credit</TableCell>
                    <TableCell>Timeslot</TableCell>
                    <TableCell>Location</TableCell>
                    <TableCell>Faculty</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {currentResult.schedule.map((row, rowIndex) => (
                    <TableRow key={rowIndex}>
                      <TableCell>{row.section}</TableCell>
                      <TableCell>{row.title}</TableCell>
                      <TableCell>{row.minCredit}</TableCell>
                      <TableCell>{row.timeslot}</TableCell>
                      <TableCell>{`${row.bldg} ${row.room}`}</TableCell>
                      <TableCell>{row.faculty1}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
         
         
          <ResultsCalendar events={currentResult.calendar_events} />
        </div>
        
      </div>
    </>
  );
  

  
};

export default ResultsTable;







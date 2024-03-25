import React, { useState } from 'react';
import Button from '@mui/material/Button';
import { Box, Typography } from '@mui/material';
import { toast } from 'react-toastify';
import '../index.css';

import PropTypes from 'prop-types'; // Import PropTypes for prop type checking


 // Function to clean data
const cleanData = (dataArray) => {
    return dataArray.map(item => {
      const cleanedItem = {};
      Object.keys(item).forEach(key => {
        const cleanKey = key.replace(/^\uFEFF/, ''); // Regex to remove BOM from start of string
        cleanedItem[cleanKey] = item[key];
      });
      return cleanedItem;
    });
};

  


  
const FileUpload = ({ onSuccess }) => {
    const [file, setFile] = useState(null);
    const [results, setResults] = useState(null); // State to hold the results
    const [isUploaded, setIsUploaded] = useState(false); // State to track if upload is done

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const uploadFileToServer = async (file) => {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://127.0.0.1:5001/load_original', {
                method: 'POST',
                body: formData,
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const result = await response.json();

            // Check if there's an error unrelated to missing columns
            if (result.error && !result.error.startsWith("Missing columns")) {
                throw new Error(result.error);
            }

            // Proceed with the assumption that the data is now correct
            const cleanedData = cleanData(result.data); // Clean the data
            // Create an object with both cleaned data and file info
            const dataWithFileInfo = {
                cleanedData,
                fileInfo: {
                    name: file.name,
                    size: file.size,
                    type: file.type || 'Unknown',
                },
            };

            setResults(cleanedData); // Update the results state with the fetched data
            onSuccess(dataWithFileInfo); // Call the onSuccess prop with the cleaned data
            setIsUploaded(true); // Set to true after successful upload
            toast.success(result.message || "File successfully uploaded and analyzed!");
       
        } catch (error) {
            console.error('Error uploading file:', error);
            toast.error(`Error uploading file: ${error.message}`);
        }
    };


    const handleUpload = () => {
        if (file) {
            uploadFileToServer(file);
            
        }
    };

    // Conditional styling or className based on isUploaded state
    const uploadStyle = isUploaded ? { display: 'none' } : {}; // Hide when uploaded

    return (
        <Box textAlign="center" id={'up_form'} style={uploadStyle}>
            <Box>
                <input
                    accept="csv/*"
                    style={{ display: 'none' }}
                    id="raised-button-file"
                    type="file"
                    onChange={handleFileChange}
                />
                <label htmlFor="raised-button-file">
                    <Button variant="contained" component="span" color="primary">
                        Choose Schedule File
                    </Button>
                </label>
                <Button
                    variant="contained"
                    color="primary"
                    onClick={handleUpload}
                    sx={{ ml: 2 }}
                    disabled={!file}
                >
                    Upload & Analyze
                </Button>
            </Box>
            {file && (
                <Box mt={2}>
                    <Typography variant="subtitle1">
                        <strong>File chosen:</strong> {file.name}
                    </Typography>
                    <Typography variant="subtitle1">
                        <strong>File size:</strong> {(file.size / 1024).toFixed(2)} KB
                    </Typography>
                    <Typography variant="subtitle1">
                        <strong>File type:</strong> {file.type || 'Unknown'}
                    </Typography>
                </Box>
            )}
            {results && (
               <div></div>
            )}
        </Box>
    );
};

FileUpload.propTypes = {
    onSuccess: PropTypes.func.isRequired,
};
export default FileUpload;

// OptimizationContext.js
import React from 'react';

const OptimizationContext = React.createContext({
  triggerOptimization: () => {},
  setTriggerOptimization: () => {},
});

export default OptimizationContext;

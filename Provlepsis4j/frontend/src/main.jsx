import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import { GraphProvider } from './graphContext.jsx'
// createRoot(document.getElementById('root')).render(
//   <GraphProvider>
//     <App />
//   </GraphProvider>
// )

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <GraphProvider>
      <App />
    </GraphProvider>
  </React.StrictMode>
);
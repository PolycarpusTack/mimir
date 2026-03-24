import React from 'react';
import ReactDOM from 'react-dom/client';
import { App } from './App';

// Global reset styles
const style = document.createElement('style');
style.textContent = `
  *, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }
  html, body, #root {
    height: 100%;
    background-color: #0d1117;
    color: #d0d0d0;
    font-family: 'Syne', sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
  ::-webkit-scrollbar {
    width: 6px;
    height: 6px;
  }
  ::-webkit-scrollbar-track {
    background: #0a0e14;
  }
  ::-webkit-scrollbar-thumb {
    background: #1e2530;
    border-radius: 3px;
  }
  ::-webkit-scrollbar-thumb:hover {
    background: #2a3545;
  }
`;
document.head.appendChild(style);

async function enableMocking() {
  if (import.meta.env.DEV && import.meta.env.VITE_MSW === 'true') {
    const { worker } = await import('../tests/mocks/browser');
    return worker.start({ onUnhandledRequest: 'bypass' });
  }
  return Promise.resolve();
}

enableMocking().then(() => {
  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
});

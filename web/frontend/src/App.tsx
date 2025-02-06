import { useState, useCallback } from 'react';
import { ThemeProvider, CssBaseline, Container, Box, Snackbar, Alert } from '@mui/material';
import theme from './theme';
import SearchBox from './components/SearchBox';
import ResearchProgress from './components/ResearchProgress';
import ResearchReport from './components/ResearchReport';
import { v4 as uuidv4 } from 'uuid';

// Backend configuration
const BACKEND_PORT = 8993;
const HOST = window.location.hostname;
const BACKEND_URL = `http://${HOST}:${BACKEND_PORT}`;
const WS_URL = `ws://${HOST}:${BACKEND_PORT}`;

function App() {
  const [clientId] = useState(() => uuidv4());
  const [isResearching, setIsResearching] = useState(false);
  const [logs, setLogs] = useState<Array<{ timestamp: string; level: string; message: string }>>([]);
  const [report, setReport] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = useCallback(async (query: string) => {
    setIsResearching(true);
    setReport(null);
    setLogs([]);
    setError(null);

    let ws: WebSocket | null = null;

    try {
      // Connect to WebSocket
      ws = new WebSocket(`${WS_URL}/ws/${clientId}`);
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'log') {
          setLogs(prev => [...prev, data.data]);
        }
      };

      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError('WebSocket connection error');
      };

      // Wait for WebSocket connection
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('WebSocket connection timeout')), 5000);
        ws!.onopen = () => {
          clearTimeout(timeout);
          resolve();
        };
      });

      const response = await fetch(`${BACKEND_URL}/api/research`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query, client_id: clientId }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      if (result.status === 'success') {
        setReport(result.result);
      } else {
        throw new Error(result.message || 'Research failed');
      }
    } catch (error) {
      console.error('Error during research:', error);
      setError(error instanceof Error ? error.message : 'An unknown error occurred');
    } finally {
      setIsResearching(false);
      if (ws) {
        ws.close();
      }
    }
  }, [clientId]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg">
        <Box
          sx={{
            minHeight: '100vh',
            py: 4,
            display: 'flex',
            flexDirection: 'column',
            gap: 4,
          }}
        >
          <SearchBox onSearch={handleSearch} disabled={isResearching} />
          <ResearchProgress logs={logs} visible={isResearching || logs.length > 0} />
          {report && <ResearchReport content={report} />}
          <Snackbar 
            open={!!error} 
            autoHideDuration={6000} 
            onClose={() => setError(null)}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
          >
            <Alert onClose={() => setError(null)} severity="error" sx={{ width: '100%' }}>
              {error}
            </Alert>
          </Snackbar>
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;

import { useState, useCallback, useRef, useEffect } from 'react';
import { ThemeProvider, CssBaseline, Container, Box, Snackbar, Alert } from '@mui/material';
import theme from './theme';
import SearchBox from './components/SearchBox';
import ResearchProgress from './components/ResearchProgress';
import ResearchReport from './components/ResearchReport';
import FollowupChat from './components/FollowupChat';
import { v4 as uuidv4 } from 'uuid';

// Backend configuration
const BACKEND_PORT = 8993;
const HOST = window.location.hostname;
const BACKEND_URL = `http://${HOST}:${BACKEND_PORT}`;
const WS_URL = `ws://${HOST}:${BACKEND_PORT}`;

interface ChatMessage {
  question: string;
  answer: string;
  timestamp: string;
}

interface GlobalState {
  is_researching: boolean;
  current_query: string | null;
  current_report: string | null;
  logs: Array<{ timestamp: string; level: string; message: string }>;
  chat_history: ChatMessage[];
}

function App() {
  const [logs, setLogs] = useState<Array<{ timestamp: string; level: string; message: string }>>([]);
  const [isResearching, setIsResearching] = useState(false);
  const [report, setReport] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentQuery, setCurrentQuery] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [chatExpanded, setChatExpanded] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const clientId = useRef<string>(uuidv4());

  // Function to update state from server state
  const updateFromGlobalState = useCallback((state: GlobalState) => {
    setIsResearching(state.is_researching);
    setCurrentQuery(state.current_query);
    setReport(state.current_report);
    setLogs(state.logs);
    if (state.chat_history) {
      setChatHistory(state.chat_history);
    }
  }, []);

  // Initialize WebSocket connection and fetch initial state
  useEffect(() => {
    const initializeState = async () => {
      try {
        setIsLoading(true);
        // Fetch initial state
        const response = await fetch(`${BACKEND_URL}/api/state`);
        if (!response.ok) {
          throw new Error('Failed to fetch initial state');
        }
        const state = await response.json();
        updateFromGlobalState(state);
        setIsLoading(false);

        // Set up WebSocket connection
        const ws = new WebSocket(`${WS_URL}/ws/${clientId.current}`);
        wsRef.current = ws;

        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.type === 'state_update') {
            updateFromGlobalState(data.data);
          } else if (data.type === 'log') {
            setLogs(prev => [...prev, data.data]);
          }
        };

        ws.onerror = () => {
          console.error('WebSocket error');
          setError('WebSocket connection error');
          setIsLoading(false);
        };

        ws.onclose = () => {
          if (wsRef.current === ws) {
            wsRef.current = null;
            // Try to reconnect after a delay
            setTimeout(initializeState, 5000);
          }
        };
      } catch (error) {
        console.error('Error:', error);
        setError(error instanceof Error ? error.message : 'An unknown error occurred');
        setIsLoading(false);
        // Try to reconnect after a delay
        setTimeout(initializeState, 5000);
      }
    };

    initializeState();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [updateFromGlobalState]);

  const handleSearch = useCallback(async (query: string) => {
    setError(null);

    try {
      const response = await fetch(`${BACKEND_URL}/api/research`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          client_id: clientId.current,
        }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.message || 'Failed to start research');
      }
    } catch (error) {
      console.error('Error:', error);
      setError(error instanceof Error ? error.message : 'An unknown error occurred');
    }
  }, []);

  const handleChatToggle = useCallback(() => {
    setChatExpanded(prev => !prev);
  }, []);

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
          <SearchBox
            onSearch={handleSearch}
            disabled={isResearching}
            isResearching={isResearching}
            currentQuery={currentQuery}
          />
          <ResearchProgress 
            logs={logs} 
            visible={isResearching && !report} 
          />
          {report && (
            <ResearchReport content={report} />
          )}
          {report && (
            <FollowupChat 
              clientId={clientId.current}
              chatHistory={chatHistory}
              isExpanded={chatExpanded}
              onToggle={handleChatToggle}
            />
          )}
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

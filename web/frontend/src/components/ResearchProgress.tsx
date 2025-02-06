import { Paper, Box, Typography, Fade } from '@mui/material';
import { useEffect, useRef } from 'react';

interface Log {
  timestamp: string;
  level: string;
  message: string;
}

interface ResearchProgressProps {
  logs: Log[];
  visible: boolean;
}

const ResearchProgress = ({ logs, visible }: ResearchProgressProps) => {
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  const getLevelColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'info':
        return '#03DAC6';
      case 'warning':
        return '#FFB74D';
      case 'error':
        return '#CF6679';
      case 'debug':
        return '#BB86FC';
      default:
        return '#FFFFFF';
    }
  };

  if (!visible) return null;

  return (
    <Fade in={visible}>
      <Paper
        elevation={3}
        sx={{
          p: 3,
          maxHeight: '400px',
          overflow: 'auto',
          backgroundColor: 'background.paper',
          borderRadius: 2,
          '&::-webkit-scrollbar': {
            width: '8px',
          },
          '&::-webkit-scrollbar-track': {
            background: 'rgba(255, 255, 255, 0.1)',
            borderRadius: '4px',
          },
          '&::-webkit-scrollbar-thumb': {
            background: 'rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            '&:hover': {
              background: 'rgba(255, 255, 255, 0.3)',
            },
          },
        }}
      >
        <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>
          Research Progress
        </Typography>
        <Box sx={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
          {logs.map((log, index) => (
            <Box
              key={index}
              sx={{
                mb: 1,
                display: 'flex',
                alignItems: 'flex-start',
                gap: 2,
                fontSize: '0.9rem',
                opacity: 0.9,
                '&:hover': {
                  opacity: 1,
                },
              }}
            >
              <Typography
                component="span"
                sx={{
                  color: 'text.secondary',
                  fontSize: '0.8rem',
                  minWidth: '160px',
                }}
              >
                {new Date(log.timestamp).toLocaleTimeString()}
              </Typography>
              <Typography
                component="span"
                sx={{
                  color: getLevelColor(log.level),
                  textTransform: 'uppercase',
                  fontSize: '0.8rem',
                  minWidth: '70px',
                }}
              >
                {log.level}
              </Typography>
              <Typography
                component="span"
                sx={{
                  color: 'text.primary',
                  flex: 1,
                }}
              >
                {log.message}
              </Typography>
            </Box>
          ))}
          <div ref={logsEndRef} />
        </Box>
      </Paper>
    </Fade>
  );
};

export default ResearchProgress; 
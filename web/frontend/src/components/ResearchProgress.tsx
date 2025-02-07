import { Paper, Box, Typography, Fade, Collapse, IconButton, Skeleton } from '@mui/material';
import { useEffect, useRef, useState, useMemo } from 'react';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { styled, keyframes } from '@mui/material/styles';

interface Log {
  timestamp: string;
  level: string;
  message: string;
}

interface ResearchProgressProps {
  logs: Log[];
  visible: boolean;
}

interface LogGroup {
  title: string;
  logs: Log[];
  type: 'search' | 'analysis' | 'extraction' | 'report' | 'other';
}

const shimmer = keyframes`
  0% {
    background-position: -1000px 0;
  }
  100% {
    background-position: 1000px 0;
  }
`;

const LoadingBox = styled(Box)(({ theme }) => ({
  animation: `${shimmer} 2s infinite linear`,
  background: `linear-gradient(to right, ${theme.palette.background.paper} 8%, ${theme.palette.action.hover} 18%, ${theme.palette.background.paper} 33%)`,
  backgroundSize: '2000px 100%',
}));

const ExpandMore = styled((props: {
  expand: boolean;
  onClick: () => void;
  children?: React.ReactNode;
}) => {
  const { expand, ...other } = props;
  return <IconButton {...other}>{props.children}</IconButton>;
})(({ theme, expand }) => ({
  transform: !expand ? 'rotate(0deg)' : 'rotate(180deg)',
  marginLeft: 'auto',
  transition: theme.transitions.create('transform', {
    duration: theme.transitions.duration.shortest,
  }),
  padding: 4,
}));

const LogGroupBox = styled(Box, {
  shouldForwardProp: (prop) => prop !== 'isActive',
})<{ isActive?: boolean }>(({ theme, isActive }) => ({
  marginBottom: theme.spacing(2),
  backgroundColor: 'rgba(255, 255, 255, 0.05)',
  borderRadius: theme.shape.borderRadius,
  overflow: 'hidden',
  transition: theme.transitions.create(['background-color', 'box-shadow'], {
    duration: theme.transitions.duration.short,
  }),
  '&:hover': {
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
  },
  ...(isActive && {
    animation: `${shimmer} 2s infinite linear`,
    background: (theme) => `linear-gradient(to right, 
      ${theme.palette.background.paper} 8%, 
      rgba(255, 255, 255, 0.1) 18%, 
      ${theme.palette.background.paper} 33%
    )`,
    backgroundSize: '2000px 100%',
  }),
}));

const LogGroupHeader = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  padding: theme.spacing(1.5, 2),
  cursor: 'pointer',
  '&:hover': {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
  },
}));

const getGroupType = (message: string): 'search' | 'analysis' | 'extraction' | 'report' | 'other' => {
  const lowerMessage = message.toLowerCase();
  if (lowerMessage.includes('search')) return 'search';
  if (lowerMessage.includes('analy')) return 'analysis';
  if (lowerMessage.includes('extract')) return 'extraction';
  if (lowerMessage.includes('report')) return 'report';
  return 'other';
};

const getGroupIcon = (type: string) => {
  switch (type) {
    case 'search':
      return '🔍';
    case 'analysis':
      return '📊';
    case 'extraction':
      return '📥';
    case 'report':
      return '📝';
    default:
      return '📌';
  }
};

const ResearchProgress = ({ logs, visible }: ResearchProgressProps) => {
  const [expandedGroups, setExpandedGroups] = useState<Set<number>>(new Set());
  const logsEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true);
  const lastUserScrollPosition = useRef<number>(0);
  const lastContentHeight = useRef<number>(0);
  const isAutoExpanding = useRef(false);

  const logGroups = useMemo(() => {
    const groups: LogGroup[] = [];
    let currentGroup: LogGroup | null = null;

    logs.forEach((log) => {
      const type = getGroupType(log.message);
      
      // Start a new group if message indicates a new operation
      if (!currentGroup || currentGroup.type !== type) {
        if (currentGroup) {
          groups.push(currentGroup);
        }
        currentGroup = {
          title: log.message,
          logs: [log],
          type,
        };
      } else {
        currentGroup.logs.push(log);
      }
    });

    if (currentGroup) {
      groups.push(currentGroup);
    }

    return groups;
  }, [logs]);

  // Auto-expand latest group when it changes
  useEffect(() => {
    if (logGroups.length > 0) {
      isAutoExpanding.current = true;
      setExpandedGroups(prev => {
        const next = new Set(prev);
        next.add(logGroups.length - 1);
        return next;
      });
    }
  }, [logGroups.length]);

  useEffect(() => {
    if (scrollContainerRef.current) {
      const container = scrollContainerRef.current;
      const { scrollHeight, clientHeight } = container;
      
      // If content height has changed and we're in auto-scroll mode or it was triggered by auto-expand
      if ((shouldAutoScroll || isAutoExpanding.current) && scrollHeight !== lastContentHeight.current) {
        container.scrollTo({
          top: scrollHeight,
          behavior: isAutoExpanding.current ? 'smooth' : 'auto'
        });
        isAutoExpanding.current = false;
      }
      
      lastContentHeight.current = scrollHeight;
    }
  }, [logs, shouldAutoScroll, expandedGroups]);

  const handleScroll = () => {
    if (scrollContainerRef.current) {
      const { scrollHeight, clientHeight, scrollTop } = scrollContainerRef.current;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
      
      // Only update shouldAutoScroll if the user has actually scrolled
      // (not when the content has changed)
      if (Math.abs(scrollTop - lastUserScrollPosition.current) > 10) {
        setShouldAutoScroll(isNearBottom);
        lastUserScrollPosition.current = scrollTop;
      }
    }
  };

  const toggleGroup = (index: number) => {
    isAutoExpanding.current = false; // Manual toggle
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

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
          maxHeight: '60vh',
          overflow: 'auto',
          backgroundColor: 'background.paper',
          borderRadius: 2,
          transition: 'all 0.3s ease-in-out',
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
        ref={scrollContainerRef}
        onScroll={handleScroll}
      >
        <Typography variant="h6" gutterBottom sx={{ color: 'primary.main', mb: 3 }}>
          Research Progress
        </Typography>
        
        {logGroups.map((group, groupIndex) => (
          <Fade in={true} key={groupIndex} timeout={300}>
            <LogGroupBox isActive={groupIndex === logGroups.length - 1}>
              <LogGroupHeader onClick={() => toggleGroup(groupIndex)}>
                <Typography
                  variant="subtitle1"
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    color: 'primary.light',
                  }}
                >
                  {group.title}
                </Typography>
                <ExpandMore
                  expand={expandedGroups.has(groupIndex)}
                  onClick={() => toggleGroup(groupIndex)}
                  aria-expanded={expandedGroups.has(groupIndex)}
                  aria-label="show more"
                >
                  <ExpandMoreIcon />
                </ExpandMore>
              </LogGroupHeader>
              
              <Collapse in={expandedGroups.has(groupIndex)} timeout="auto">
                <Box sx={{ p: 2, pt: 0 }}>
                  {group.logs.map((log, logIndex) => (
                    <Fade in={true} key={logIndex} timeout={300}>
                      <Box
                        sx={{
                          py: 0.5,
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
                            minWidth: '100px',
                            fontFamily: 'Roboto Mono, monospace',
                          }}
                        >
                          {new Date(log.timestamp).toLocaleTimeString()}
                        </Typography>
                        <Typography
                          component="span"
                          sx={{
                            color: getLevelColor(log.level),
                            fontSize: '0.8rem',
                            minWidth: '60px',
                            textTransform: 'uppercase',
                            fontFamily: 'Roboto Mono, monospace',
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
                    </Fade>
                  ))}
                </Box>
              </Collapse>
            </LogGroupBox>
          </Fade>
        ))}
        <div ref={logsEndRef} />
      </Paper>
    </Fade>
  );
};

export default ResearchProgress; 
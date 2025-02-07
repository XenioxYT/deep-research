import { Paper, Box, Typography, Fade, Collapse, IconButton } from '@mui/material';
import { useEffect, useRef, useState, useMemo } from 'react';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { styled, keyframes, Theme, useTheme } from '@mui/material/styles';

interface Log {
  timestamp: string;
  level: string;
  message: string;
}

interface ResearchProgressProps {
  logs: Log[];
  visible: boolean;
}

type GroupType = 'search' | 'analysis' | 'extraction' | 'report' | 'other';

interface LogGroup {
  title: string;
  logs: Log[];
  type: GroupType;
}

const shimmer = keyframes`
  0% {
    background-position: -1000px 0;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0);
  }
  50% {
    box-shadow: 0 0 20px rgba(187, 134, 252, 0.25);
  }
  100% {
    background-position: 1000px 0;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0);
  }
`;

const glow = keyframes`
  0% {
    box-shadow: 0 0 10px rgba(187, 134, 252, 0.125);
  }
  50% {
    box-shadow: 0 0 20px rgba(187, 134, 252, 0.25);
  }
  100% {
    box-shadow: 0 0 10px rgba(187, 134, 252, 0.125);
  }
`;

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

const LogGroupBox = styled(Box)<{ isActive?: boolean }>(({ theme, isActive }) => ({
  marginBottom: theme.spacing(2),
  backgroundColor: theme.palette.background.paper,
  borderRadius: theme.shape.borderRadius,
  overflow: 'hidden',
  border: `1px solid ${isActive ? theme.palette.primary.main : theme.palette.divider}`,
  transition: theme.transitions.create(['background-color', 'box-shadow', 'border-color'], {
    duration: theme.transitions.duration.short,
  }),
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
    borderColor: theme.palette.primary.main,
  },
  ...(isActive && {
    borderColor: theme.palette.primary.main,
    animation: `${glow} 2s infinite ease-in-out, ${shimmer} 3s infinite linear`,
    background: `linear-gradient(to right, 
      ${theme.palette.background.paper} 0%,
      ${theme.palette.action.hover} 20%,
      ${theme.palette.background.paper} 40%
    )`,
    backgroundSize: '1000px 100%',
  }),
}));

const LogGroupHeader = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  padding: theme.spacing(1.5, 2),
  cursor: 'pointer',
  transition: theme.transitions.create(['background-color', 'box-shadow'], {
    duration: theme.transitions.duration.short,
  }),
  '&:hover': {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    boxShadow: '0 0 20px rgba(187, 134, 252, 0.1)',
  },
}));

const getGroupType = (message: string): GroupType => {
  const lowerMessage = message.toLowerCase();
  if (lowerMessage.includes('starting research')) return 'search';
  if (lowerMessage.includes('analyzing')) return 'analysis';
  if (lowerMessage.includes('extracting')) return 'extraction';
  if (lowerMessage.includes('report')) return 'report';
  if (lowerMessage.includes('iteration')) return 'search';
  if (lowerMessage.includes('processing')) return 'search';
  if (lowerMessage.includes('batch searching')) return 'search';
  if (lowerMessage.includes('ranking')) return 'analysis';
  if (lowerMessage.includes('scraping')) return 'extraction';
  if (lowerMessage.includes('research complete') || lowerMessage.includes('moving to final report')) return 'report';
  return 'other';
};

const getLevelColor = (level: string, theme: Theme) => {
  switch (level.toLowerCase()) {
    case 'info':
      return theme.palette.info.main;
    case 'warning':
      return theme.palette.warning.main;
    case 'error':
      return theme.palette.error.main;
    case 'debug':
      return theme.palette.primary.main;
    default:
      return theme.palette.text.primary;
  }
};

const getGroupTitle = (message: string): string => {
  const lowerMessage = message.toLowerCase();
  if (lowerMessage.includes('starting research')) {
    return 'Starting Research';
  }
  if (lowerMessage.includes('iteration')) {
    const match = message.match(/Iteration (\d+)/);
    return match ? `Research Iteration ${match[1]}` : 'Research Iteration';
  }
  if (lowerMessage.includes('analyzing')) {
    return 'Analysis Phase';
  }
  if (lowerMessage.includes('generating')) {
    return 'Report Generation';
  }
  if (lowerMessage.includes('batch searching')) {
    return 'Search Phase';
  }
  if (lowerMessage.includes('ranking')) {
    return 'Result Ranking';
  }
  if (lowerMessage.includes('scraping')) {
    return 'Content Extraction';
  }
  if (lowerMessage.includes('research complete') || lowerMessage.includes('moving to final report')) {
    return 'Research Complete';
  }
  return 'Other Operations';
};

interface MessageGroup {
  type: 'extraction' | 'other';
  title: string;
  messages: string[];
}

const ResearchProgress = ({ logs, visible }: ResearchProgressProps) => {
  const theme = useTheme();
  const [expandedGroups, setExpandedGroups] = useState<Set<number>>(new Set());
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const lastUserScrollPosition = useRef<number>(0);
  const lastLogsLength = useRef<number>(0);
  const extractionStartTime = useRef<number | null>(null);
  const [extractionDuration, setExtractionDuration] = useState<number>(0);
  const [showWritingReport, setShowWritingReport] = useState(false);
  const extractionTimerRef = useRef<ReturnType<typeof setInterval>>(null);
  const [messageGroups, setMessageGroups] = useState<MessageGroup[]>([]);

  // Track extraction time
  useEffect((): (() => void) => {
    const cleanup = () => {
      if (extractionTimerRef.current) {
        clearInterval(extractionTimerRef.current);
      }
    };

    const lastLog = logs[logs.length - 1];
    if (!lastLog) {
      cleanup();
      return cleanup;
    }

    const message = lastLog.message.toLowerCase();
    
    // Start timer when extraction begins
    if (message.includes('extracting content') || message.includes('scraping')) {
      if (!extractionStartTime.current) {
        extractionStartTime.current = Date.now();
        setShowWritingReport(false);
        
        // Start interval to update duration (hidden from display)
        extractionTimerRef.current = setInterval(() => {
          const duration = Math.floor((Date.now() - (extractionStartTime.current || 0)) / 1000);
          setExtractionDuration(duration);
          
          // Show writing report after 5 seconds
          if (duration > 5) {
            setShowWritingReport(true);
          }
        }, 1000);
      }
    } else if (!message.includes('extracting') && !message.includes('scraping')) {
      // Reset timer if we're not extracting
      extractionStartTime.current = null;
      setExtractionDuration(0);
      setShowWritingReport(false);
      cleanup();
    }

    return cleanup;
  }, [logs]);

  const getExtractionTitle = (baseTitle: string): string => {
    return baseTitle;  // Remove duration from title
  };

  const logGroups = useMemo(() => {
    const groups: LogGroup[] = [];
    let currentGroup: LogGroup | null = null;

    const shouldStartNewGroup = (message: string): boolean => {
      const lowerMessage = message.toLowerCase();
      return lowerMessage.includes('starting research') || 
             lowerMessage.includes('iteration') ||
             lowerMessage.includes('analyzing') ||
             lowerMessage.includes('generating') ||
             lowerMessage.includes('batch searching') ||
             lowerMessage.includes('ranking') ||
             lowerMessage.includes('scraping') ||
             lowerMessage.includes('research complete') ||
             lowerMessage.includes('moving to final report');
    };

    const mergeGroups = (inputGroups: LogGroup[]): LogGroup[] => {
      const merged: LogGroup[] = [];
      let current: LogGroup | null = null;

      inputGroups.forEach((group) => {
        // Special handling for research complete messages
        if (group.logs.some(log => 
          log.message.toLowerCase().includes('research complete') || 
          log.message.toLowerCase().includes('moving to final report')
        )) {
          if (current) {
            merged.push(current);
          }
          const reportGroup: LogGroup = {
            title: 'Research Complete, Writing Report',
            logs: group.logs,
            type: 'report'
          };
          merged.push(reportGroup);
          current = null;
        } else if (!current || current.type !== group.type) {
          if (current) {
            merged.push(current);
          }
          current = { ...group };
        } else {
          // Merge logs into the current group
          current.logs.push(...group.logs);
          
          // Update title if it's an iteration
          if (current.title.includes('Research Iteration')) {
            const match = group.title.match(/Iteration (\d+)/);
            if (match) {
              current.title = `Research Iteration ${match[1]}`;
            }
          }
        }
      });

      if (current) {
        merged.push(current);
      }

      return merged;
    };

    // First pass: create initial groups
    logs.forEach((log) => {
      const type = getGroupType(log.message);
      const isNewOperation = shouldStartNewGroup(log.message);
      
      if (!currentGroup || isNewOperation) {
        if (currentGroup) {
          groups.push(currentGroup);
        }
        let title = getGroupTitle(log.message);
        // Add duration to extraction title
        if (type === 'extraction') {
          title = getExtractionTitle(title);
        }
        const newGroup: LogGroup = {
          title,
          logs: [log],
          type,
        };
        currentGroup = newGroup;
      } else {
        currentGroup.logs.push(log);
        // Update extraction title with duration
        if (currentGroup.type === 'extraction') {
          currentGroup.title = getExtractionTitle(getGroupTitle(log.message));
        }
      }
    });

    if (currentGroup) {
      // Update final group title if it's extraction
      if (currentGroup.type === 'extraction') {
        currentGroup.title = getExtractionTitle(currentGroup.title);
      }
      groups.push(currentGroup);
    }

    // Second pass: merge consecutive groups of the same type
    const mergedGroups = mergeGroups(groups);

    // Add writing report group if needed
    if (showWritingReport && !mergedGroups.some((g: LogGroup) => g.title.includes('Writing Report'))) {
      const reportGroup: LogGroup = {
        title: 'Writing Research Report',
        logs: [{
          timestamp: new Date().toISOString(),
          level: 'info',
          message: 'Generating final research report...'
        }],
        type: 'report'
      };
      mergedGroups.push(reportGroup);
    }

    return mergedGroups;
  }, [logs, extractionDuration, showWritingReport]);

  // Track when logs change
  useEffect(() => {
    if (logs.length !== lastLogsLength.current) {
      lastLogsLength.current = logs.length;
      
      // Ensure we're at the bottom after new logs are added
      if (scrollContainerRef.current) {
        const container = scrollContainerRef.current;
        
        // First timeout: wait for the new content to be rendered
        setTimeout(() => {
          // Second timeout: wait for the expansion animation
          setTimeout(() => {
            container.scrollTo({
              top: container.scrollHeight,
              behavior: 'smooth'
            });
          }, 300);
        }, 0);
      }
    }
  }, [logs]);

  // Auto-expand latest group when logs change
  useEffect(() => {
    if (logGroups.length > 0) {
      setExpandedGroups(new Set([logGroups.length - 1]));
      
      // Ensure we scroll after expanding
      if (scrollContainerRef.current) {
        const container = scrollContainerRef.current;
        
        // Wait for both content update and expansion animation
        setTimeout(() => {
          container.scrollTo({
            top: container.scrollHeight,
            behavior: 'smooth'
          });
        }, 350); // Slightly longer than the animation to ensure completion
      }
    }
  }, [logGroups.length]);

  // Handle scroll position changes
  const handleScroll = () => {
    if (scrollContainerRef.current) {
      const { scrollHeight, clientHeight, scrollTop } = scrollContainerRef.current;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
      
      if (Math.abs(scrollTop - lastUserScrollPosition.current) > 10) {
        lastUserScrollPosition.current = scrollTop;
      }
    }
  };

  const toggleGroup = (index: number) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
        // Scroll to the expanded group after animation
        setTimeout(() => {
          const element = document.getElementById(`log-group-${index}`);
          if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          }
        }, 300);
      }
      return next;
    });
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
          scrollBehavior: 'smooth',
          '&::-webkit-scrollbar': {
            width: '8px',
          },
          '&::-webkit-scrollbar-track': {
            background: theme.palette.action.hover,
            borderRadius: '4px',
          },
          '&::-webkit-scrollbar-thumb': {
            background: theme.palette.primary.main,
            borderRadius: '4px',
            opacity: 0.7,
            '&:hover': {
              opacity: 1,
            },
          },
        }}
        ref={scrollContainerRef}
        onScroll={handleScroll}
      >
        <Typography variant="h6" gutterBottom sx={{ color: 'primary.main', mb: 3, fontWeight: 500 }}>
          Research Progress
        </Typography>
        
        {logGroups.map((group, groupIndex) => (
          <Fade in={true} key={groupIndex} timeout={300}>
            <LogGroupBox 
              id={`log-group-${groupIndex}`}
              isActive={groupIndex === logGroups.length - 1}
            >
              <LogGroupHeader onClick={() => toggleGroup(groupIndex)}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Typography
                    variant="subtitle1"
                    sx={{
                      color: 'text.primary',
                      fontWeight: 500,
                    }}
                  >
                    {group.title}
                  </Typography>
                  <Typography
                    variant="caption"
                    sx={{
                      color: 'text.secondary',
                      fontFamily: 'Roboto Mono, monospace',
                    }}
                  >
                    {group.logs.length} entries
                  </Typography>
                </Box>
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
                          borderBottom: logIndex < group.logs.length - 1 ? `1px solid ${theme.palette.divider}` : 'none',
                          '&:hover': {
                            backgroundColor: theme.palette.action.hover,
                          },
                        }}
                      >
                        <Typography
                          component="span"
                          sx={{
                            color: 'text.secondary',
                            fontSize: '0.8rem',
                            minWidth: '80px',
                            fontFamily: 'Roboto Mono, monospace',
                          }}
                        >
                          {new Date(log.timestamp).toLocaleTimeString()}
                        </Typography>
                        <Typography
                          component="span"
                          sx={{
                            color: getLevelColor(log.level, theme),
                            fontSize: '0.8rem',
                            minWidth: '60px',
                            textTransform: 'uppercase',
                            fontFamily: 'Roboto Mono, monospace',
                            fontWeight: 500,
                          }}
                        >
                          {log.level}
                        </Typography>
                        <Typography
                          component="span"
                          sx={{
                            color: 'text.primary',
                            flex: 1,
                            fontFamily: 'Roboto Mono, monospace',
                            fontSize: '0.85rem',
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-word',
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
      </Paper>
    </Fade>
  );
};

export default ResearchProgress; 
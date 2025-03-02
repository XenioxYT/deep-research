import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  TextField,
  IconButton,
  Typography,
  Collapse,
  Fade,
  CircularProgress,
  Fab,
  Zoom,
  Divider,
  Avatar,
  Tooltip,
  useTheme,
  Card,
  CardContent,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import ChatIcon from '@mui/icons-material/Chat';
import CloseIcon from '@mui/icons-material/Close';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import OpenInFullIcon from '@mui/icons-material/OpenInFull';
import CloseFullscreenIcon from '@mui/icons-material/CloseFullscreen';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { styled } from '@mui/material/styles';
import QuestionAnswerIcon from '@mui/icons-material/QuestionAnswer';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import FullscreenExitIcon from '@mui/icons-material/FullscreenExit';

const BACKEND_PORT = 8993;
const HOST = window.location.hostname;
const BACKEND_URL = `http://${HOST}:${BACKEND_PORT}`;

interface ChatMessage {
  question: string;
  answer: string;
  timestamp: string;
}

interface SourceTooltipData {
  url: string;
  title: string;
  domain: string;
}

interface Source {
  id: string;
  title: string;
  url: string;
  domain: string;
  score: number;
}

interface FollowupChatProps {
  clientId: string;
  chatHistory?: ChatMessage[];
  isExpanded?: boolean;
  onToggle?: () => void;
  reportSources?: Map<string, Source>;
}

// Add global styles for animations
const GlobalStyles = styled('style')({
  '@global': {
    '.chat-fullscreen-active': {
      overflow: 'hidden',
    },
    '@keyframes fadeIn': {
      '0%': {
        opacity: 0,
      },
      '100%': {
        opacity: 1,
      }
    },
    '@keyframes slideUp': {
      '0%': {
        transform: 'translateY(20px)',
        opacity: 0,
      },
      '100%': {
        transform: 'translateY(0)',
        opacity: 1,
      }
    },
    '@keyframes pulse': {
      '0%': {
        transform: 'scale(0.95)',
      },
      '50%': {
        transform: 'scale(1)',
      },
      '100%': {
        transform: 'scale(0.95)',
      }
    }
  }
});

const ChatContainer = styled(Paper, {
  shouldForwardProp: (prop) => prop !== 'isFullscreen'
})<{ isFullscreen?: boolean }>(({ theme, isFullscreen }) => ({
  position: 'fixed',
  bottom: theme.spacing(2),
  right: theme.spacing(2),
  width: isFullscreen ? '80vw' : '380px',
  maxWidth: isFullscreen ? '1200px' : '90vw',
  maxHeight: isFullscreen ? '70vh' : '500px',
  height: isFullscreen ? 'auto' : 'auto',
  display: 'flex',
  flexDirection: 'column',
  zIndex: 1000,
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[10],
  overflow: 'hidden',
  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  transform: 'translate3d(0, 0, 0)', // Force GPU acceleration for smoother animations
  willChange: 'transform, width, height, max-height', // Optimize animations
}));

const ChatHeader = styled(Box)(({ theme }) => ({
  backgroundColor: theme.palette.primary.main,
  color: theme.palette.primary.contrastText,
  padding: theme.spacing(1.5, 2),
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  cursor: 'pointer',
}));

const ChatMessages = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2),
  overflowY: 'auto',
  flexGrow: 1,
  height: 'auto',
  maxHeight: 'calc(100% - 120px)', // Account for header and input
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(2),
  transition: 'all 0.3s ease',
  '&::-webkit-scrollbar': {
    width: '6px',
  },
  '&::-webkit-scrollbar-track': {
    background: 'rgba(0,0,0,0.05)',
    borderRadius: '4px',
  },
  '&::-webkit-scrollbar-thumb': {
    background: theme.palette.primary.light,
    borderRadius: '4px',
  },
}));

const MessageBubble = styled(Paper, {
  shouldForwardProp: (prop) => prop !== 'isUser'
})<{ isUser: boolean }>(({ theme, isUser }) => ({
  padding: theme.spacing(1.5, 2),
  backgroundColor: isUser
    ? theme.palette.primary.light
    : theme.palette.background.default,
  color: isUser
    ? theme.palette.primary.contrastText
    : theme.palette.text.primary,
  borderRadius: isUser
    ? theme.shape.borderRadius + 'px ' + theme.shape.borderRadius + 'px 0 ' + theme.shape.borderRadius + 'px'
    : '0 ' + theme.shape.borderRadius + 'px ' + theme.shape.borderRadius + 'px ' + theme.shape.borderRadius + 'px',
  alignSelf: isUser ? 'flex-end' : 'flex-start',
  maxWidth: '85%',
  wordBreak: 'break-word',
  boxShadow: theme.shadows[1],
  border: isUser ? 'none' : `1px solid ${theme.palette.divider}`,
  animation: 'slideUp 0.3s ease',
  opacity: 1,
}));

const PendingMessageBubble = styled(MessageBubble)<{ isUser: boolean }>(({ theme }) => ({
  animation: 'pulse 2s infinite',
  opacity: 0.7,
}));

const ChatInput = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2),
  borderTop: `1px solid ${theme.palette.divider}`,
  display: 'flex',
  gap: theme.spacing(1),
  alignItems: 'center',
  position: 'relative',
  backgroundColor: theme.palette.background.paper,
  transition: 'opacity 0.3s ease',
  opacity: 1,
  '&.disabled': {
    opacity: 0.7,
  },
}));

const StyledMarkdown = styled(Box)(({ theme }) => ({
  '& .markdown-body': {
    color: theme.palette.text.primary,
    backgroundColor: 'transparent',
    fontFamily: theme.typography.fontFamily,
    fontSize: '1rem',
    lineHeight: 1.7,
    maxWidth: '100%',
    margin: '0 auto',
    
    '& h1': {
      color: theme.palette.primary.main,
      fontSize: '1.8rem',
      fontWeight: 500,
      marginBottom: theme.spacing(2),
      borderBottom: `1px solid ${theme.palette.divider}`,
      paddingBottom: theme.spacing(1),
      fontFamily: '"Google Sans", sans-serif',
    },
    
    '& h2': {
      color: theme.palette.primary.light,
      fontSize: '1.5rem',
      fontWeight: 500,
      marginTop: theme.spacing(4),
      marginBottom: theme.spacing(2),
      borderBottom: `1px solid ${theme.palette.divider}`,
      paddingBottom: theme.spacing(1),
      fontFamily: '"Google Sans", sans-serif',
    },
    
    '& h3': {
      color: theme.palette.secondary.main,
      fontSize: '1.3rem',
      fontWeight: 500,
      marginTop: theme.spacing(3),
      marginBottom: theme.spacing(2),
      fontFamily: '"Google Sans", sans-serif',
    },
    
    '& p': {
      marginBottom: theme.spacing(2),
      fontSize: '1rem',
      lineHeight: 1.7,
    },
    
    '& a': {
      color: theme.palette.primary.main,
      textDecoration: 'none',
      borderBottom: `1px solid ${theme.palette.primary.main}`,
      transition: theme.transitions.create(['border-color', 'color'], {
        duration: theme.transitions.duration.shortest,
      }),
      '&:hover': {
        color: theme.palette.primary.light,
        borderColor: theme.palette.primary.light,
      },
    },
    
    '& code': {
      backgroundColor: 'rgba(187, 134, 252, 0.1)',
      color: theme.palette.primary.light,
      padding: '2px 6px',
      borderRadius: theme.shape.borderRadius,
      fontSize: '0.9em',
      fontFamily: 'Roboto Mono, monospace',
    },
    
    '& pre': {
      backgroundColor: 'rgba(187, 134, 252, 0.05)',
      padding: theme.spacing(2),
      borderRadius: theme.shape.borderRadius,
      overflow: 'auto',
      margin: theme.spacing(2, 0),
      '& code': {
        backgroundColor: 'transparent',
        padding: 0,
        fontSize: '0.9em',
        fontFamily: 'Roboto Mono, monospace',
      },
    },
    
    '& blockquote': {
      borderLeft: `4px solid ${theme.palette.primary.main}`,
      margin: theme.spacing(2, 0),
      padding: theme.spacing(1, 2),
      backgroundColor: 'rgba(187, 134, 252, 0.05)',
      borderRadius: `0 ${theme.shape.borderRadius}px ${theme.shape.borderRadius}px 0`,
      color: theme.palette.text.secondary,
    },
    
    '& table': {
      width: '100%',
      borderCollapse: 'collapse',
      marginBottom: theme.spacing(3),
      display: 'block',
      overflowX: 'auto',
      '& thead': {
        position: 'sticky',
        top: '-1px',
        zIndex: 2,
        backgroundColor: theme.palette.background.paper,
        boxShadow: theme.shadows[1]
      },
      '& th': {
        backgroundColor: 'rgba(187, 134, 252, 0.1)',
        color: theme.palette.primary.light,
        fontWeight: 500,
        textAlign: 'left',
        padding: theme.spacing(1.5),
        borderBottom: `2px solid ${theme.palette.divider}`,
        position: 'sticky',
        top: '0',
      },
      '& td': {
        padding: theme.spacing(1.5),
        borderBottom: `1px solid ${theme.palette.divider}`,
      },
      '& tr:last-child td': {
        borderBottom: 'none',
      },
      '& tr:hover td': {
        backgroundColor: 'rgba(255, 255, 255, 0.03)',
      },
    },
    
    '& ul, & ol': {
      paddingLeft: theme.spacing(4),
      marginBottom: theme.spacing(2),
    },
    
    '& li': {
      marginBottom: theme.spacing(1),
      '&::marker': {
        color: theme.palette.primary.main,
      },
    },
    
    '& hr': {
      border: 'none',
      height: '1px',
      backgroundColor: theme.palette.divider,
      margin: theme.spacing(3, 0),
    },
    
    '& img': {
      maxWidth: '100%',
      height: 'auto',
      borderRadius: theme.shape.borderRadius,
      margin: theme.spacing(2, 0),
    },
    
    '& .katex-display': {
      margin: theme.spacing(2, 0),
      overflow: 'auto',
      '& > .katex': {
        maxWidth: '100%',
      }
    },
    
    '& .katex': {
      fontSize: '1.1em',
      fontFamily: 'KaTeX_Math',
      whiteSpace: 'nowrap',
    },
    
    '& .source-link': {
      color: theme.palette.primary.main,
      cursor: 'pointer',
      borderBottom: `1px dotted ${theme.palette.primary.main}`,
      fontWeight: 'bold',
      textDecoration: 'none',
      '&:hover': {
        backgroundColor: 'rgba(0, 0, 0, 0.05)',
        textDecoration: 'none',
      },
    },
  },
}));

const FloatingButton = styled(Fab)(({ theme }) => ({
  position: 'fixed',
  bottom: theme.spacing(4),
  right: theme.spacing(4),
  zIndex: 1000,
}));

const SourceTooltip = styled(({ className, title, children, ...props }: {
  className?: string;
  title: React.ReactElement;
  children: React.ReactElement;
  [key: string]: any;
}) => (
  <Tooltip {...props} classes={{ popper: className }} title={title}>
    {children}
  </Tooltip>
))(({ theme }) => ({
  '& .MuiTooltip-tooltip': {
    backgroundColor: 'transparent',
    padding: 0,
  },
}));

const SourceCard = styled(Card)(({ theme }) => ({
  backgroundColor: theme.palette.background.paper,
  minWidth: 200,
  maxWidth: 300,
  border: `1px solid ${theme.palette.divider}`,
  boxShadow: theme.shadows[10],
}));

const SourceBadge = styled(Box, {
  shouldForwardProp: (prop) => !['isUser', 'component'].includes(String(prop))
})<{ component?: React.ElementType }>(({ theme }) => ({
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  minWidth: '20px',
  height: '20px',
  padding: '0 4px',
  backgroundColor: 'rgba(187, 134, 252, 0.1)',
  border: `1px solid ${theme.palette.primary.main}`,
  borderRadius: '4px',
  color: theme.palette.primary.main,
  cursor: 'pointer',
  fontSize: '0.75rem',
  fontWeight: 500,
  textAlign: 'center',
  transition: theme.transitions.create(['background-color', 'border-color', 'color'], {
    duration: theme.transitions.duration.shortest,
  }),
  '&:hover': {
    backgroundColor: theme.palette.primary.main,
    color: theme.palette.common.white,
  },
  '&:active': {
    transform: 'scale(0.95)',
  },
}));

const CollapsedContainer = styled(Paper)(({ theme }) => ({
  position: 'fixed',
  bottom: theme.spacing(2),
  right: theme.spacing(2),
  padding: theme.spacing(1.5),
  borderRadius: theme.shape.borderRadius,
  display: 'flex',
  alignItems: 'center',
  cursor: 'pointer',
  zIndex: 1000,
  boxShadow: theme.shadows[5],
  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  animation: 'fadeIn 0.3s ease',
  '&:hover': {
    boxShadow: theme.shadows[8],
    transform: 'translateY(-2px)',
  },
  '&:active': {
    transform: 'translateY(0px)',
  },
}));

const Message = styled(Box, {
  shouldForwardProp: (prop) => !['isAi', 'isPending'].includes(String(prop))
})<{ isAi?: boolean; isPending?: boolean }>(({ theme, isAi, isPending }) => ({
  display: 'flex',
  flexDirection: 'row',
  alignItems: 'flex-start',
  gap: theme.spacing(1),
  maxWidth: '100%',
  animation: isPending ? 'pulse 2s infinite' : 'slideUp 0.3s ease',
  opacity: isPending ? 0.7 : 1,
  padding: theme.spacing(1),
  borderRadius: theme.shape.borderRadius,
  backgroundColor: isAi 
    ? theme.palette.mode === 'dark' 
      ? 'rgba(0, 0, 0, 0.2)' 
      : 'rgba(0, 0, 0, 0.05)'
    : 'transparent',
}));

const MessageAvatar = styled(Avatar, {
  shouldForwardProp: (prop) => prop !== 'isAi'
})<{ isAi?: boolean }>(({ theme, isAi }) => ({
  backgroundColor: isAi 
    ? theme.palette.primary.main 
    : theme.palette.secondary.main,
  width: 32,
  height: 32,
  fontSize: '0.875rem',
}));

// Define types for our chat messages
interface ChatMessageInterface {
  role: 'user' | 'assistant';
  content: string;
  isPending: boolean;
  isError?: boolean;
}

// Updated Source interface to match ResearchReport.tsx
interface Source {
  id: string;
  title: string;
  url: string;
  domain: string;
  score: number;
}

// Enhanced markdown component for the chat with improved source handling
const EnhancedMarkdown: React.FC<{ content: string; sources: Map<string, Source> }> = ({ content, sources }) => {
  const theme = useTheme();
  
  const processChildren = (children: React.ReactNode): React.ReactNode => {
    return React.Children.map(children, (child) => {
      if (typeof child === 'string') {
        // Match numeric citations [1], [2], etc.
        const parts = child.split(/(\[\d+(?:\s*,\s*\d+)*\])/g);
        return parts.map((part, i) => {
          // Check for numeric citations [1] or multiple citations [1, 2, 3]
          const numericSourceMatch = part.match(/\[(\d+(?:\s*,\s*\d+)*)\]/);
          if (numericSourceMatch) {
            const numbers = numericSourceMatch[1].split(/\s*,\s*/).map(num => num.trim());
            if (numbers.length === 1) {
              const sourceKey = `[${numbers[0]}]`;
              const source = sources.get(sourceKey);
              if (!source) return part;
              
              return (
                <Box
                  key={i}
                  component="a"
                  href={source.url || "#"}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => {
                    if (!source.url) e.preventDefault();
                  }}
                  className="source-link"
                  sx={{
                    color: 'primary.main',
                    fontWeight: 'bold',
                    textDecoration: 'none',
                    borderBottom: `1px dotted ${theme.palette.primary.main}`,
                    '&:hover': {
                      backgroundColor: 'rgba(0, 0, 0, 0.05)',
                      textDecoration: 'none',
                    },
                  }}
                >
                  {sourceKey}
                </Box>
              );
            }
            
            // Handle multiple citations [1, 2, 3]
            return (
              <Box
                key={i}
                component="span"
                sx={{ 
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 0.5,
                }}
              >
                {numbers.map((num, index) => {
                  const sourceKey = `[${num}]`;
                  const source = sources.get(sourceKey);
                  if (!source) return num;
                  
                  return (
                    <React.Fragment key={num}>
                      {index > 0 && ', '}
                      <Box
                        component="a"
                        href={source.url || "#"}
                        target="_blank"
                        rel="noopener noreferrer"
                        onClick={(e) => {
                          if (!source.url) e.preventDefault();
                        }}
                        className="source-link"
                        sx={{
                          color: 'primary.main',
                          fontWeight: 'bold',
                          textDecoration: 'none',
                          borderBottom: `1px dotted ${theme.palette.primary.main}`,
                          '&:hover': {
                            backgroundColor: 'rgba(0, 0, 0, 0.05)',
                            textDecoration: 'none',
                          },
                        }}
                      >
                        {sourceKey}
                      </Box>
                    </React.Fragment>
                  );
                })}
              </Box>
            );
          }
          
          return part;
        });
      }
      return child;
    });
  };
  
  const components = {
    p: ({ children }: any) => (
      <Typography variant="body1" paragraph>
        {processChildren(children)}
      </Typography>
    ),
    strong: ({ children }: any) => (
      <Typography component="strong" sx={{ fontWeight: 'bold', display: 'inline' }}>
        {children}
      </Typography>
    ),
    a: ({ node, href, children }: any) => (
      <Box
        component="a"
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        sx={{
          color: 'primary.main',
          textDecoration: 'none',
          borderBottom: `1px solid`,
          '&:hover': {
            color: 'primary.light',
          }
        }}
      >
        {children}
      </Box>
    ),
    h1: ({ children }: any) => (
      <Typography variant="h1" gutterBottom>
        {children}
      </Typography>
    ),
    h2: ({ children }: any) => (
      <Typography variant="h2" gutterBottom>
        {children}
      </Typography>
    ),
    h3: ({ children }: any) => (
      <Typography variant="h3" gutterBottom>
        {children}
      </Typography>
    ),
    ul: ({ children }: any) => (
      <Box component="ul" sx={{ mb: 2, pl: 4 }}>
        {children}
      </Box>
    ),
    ol: ({ children }: any) => (
      <Box component="ol" sx={{ mb: 2, pl: 4 }}>
        {children}
      </Box>
    ),
    li: ({ children }: any) => <li>{processChildren(children)}</li>,
    blockquote: ({ children }: any) => (
      <Box
        component="blockquote"
        sx={{
          borderLeft: 4,
          borderColor: 'primary.main',
          pl: 2,
          py: 1,
          my: 2,
          bgcolor: 'rgba(187, 134, 252, 0.05)',
          borderRadius: '0 4px 4px 0',
          color: 'text.secondary',
        }}
      >
        {children}
      </Box>
    ),
    code: ({ node, inline, className, children, ...props }: any) => {
      const match = /language-(\w+)/.exec(className || '');
      return !inline ? (
        <Box
          component="pre"
          sx={{
            backgroundColor: 'rgba(187, 134, 252, 0.05)',
            p: 2,
            borderRadius: 1,
            overflow: 'auto',
            my: 2,
          }}
        >
          <Box
            component="code"
            className={className}
            sx={{
              fontFamily: 'Roboto Mono, monospace',
              fontSize: '0.9em',
            }}
            {...props}
          >
            {children}
          </Box>
        </Box>
      ) : (
        <Box
          component="code"
          className={className}
          sx={{
            backgroundColor: 'rgba(187, 134, 252, 0.1)',
            color: 'primary.light',
            p: '2px 6px',
            borderRadius: 1,
            fontSize: '0.9em',
            fontFamily: 'Roboto Mono, monospace',
          }}
          {...props}
        >
          {children}
        </Box>
      );
    },
    table: ({ children }: any) => (
      <Box
        component="div"
        sx={{
          width: '100%',
          overflowX: 'auto',
          mb: 3,
        }}
      >
        <Box 
          component="table"
          sx={{
            width: '100%',
            borderCollapse: 'collapse',
          }}
        >
          {children}
        </Box>
      </Box>
    ),
    th: ({ children }: any) => (
      <Box
        component="th"
        sx={{
          backgroundColor: 'rgba(187, 134, 252, 0.1)',
          color: 'primary.light',
          fontWeight: 500,
          textAlign: 'left',
          p: 1.5,
          borderBottom: 2,
          borderColor: 'divider',
          position: 'sticky',
          top: 0,
        }}
      >
        {children}
      </Box>
    ),
    td: ({ children }: any) => (
      <Box
        component="td"
        sx={{
          p: 1.5,
          borderBottom: 1,
          borderColor: 'divider',
          '&:hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.03)',
          },
        }}
      >
        {processChildren(children)}
      </Box>
    ),
    img: ({ src, alt }: any) => (
      <Box
        component="img"
        src={src}
        alt={alt}
        sx={{
          maxWidth: '100%',
          height: 'auto',
          borderRadius: 1,
          my: 2,
        }}
      />
    ),
  };
  
  return (
    <StyledMarkdown>
      <ReactMarkdown
        className="markdown-body"
        remarkPlugins={[
          remarkGfm,
          [remarkMath, { inlineMath: [], displayMath: [['$$', '$$']] }]
        ]}
        rehypePlugins={[rehypeKatex]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </StyledMarkdown>
  );
};

// Updated MessageContent component to provide adequate space for the markdown content
const MessageContent = styled(Box)(({ theme }) => ({
  width: '100%',
}));

const FollowupChat: React.FC<FollowupChatProps> = ({ 
  clientId, 
  chatHistory: externalChatHistory = [],
  isExpanded = false, 
  onToggle = () => {},
  reportSources = new Map() // Default to empty Map if not provided
}) => {
  const [userQuestion, setUserQuestion] = useState('');
  const [expanded, setExpanded] = useState(isExpanded);
  const [loading, setLoading] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [pendingMessages, setPendingMessages] = useState<{question: string; loading: boolean; error?: boolean}[]>([]);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [minimizing, setMinimizing] = useState(false);
  const [clickPrevented, setClickPrevented] = useState(false);
  const [showCollapsed, setShowCollapsed] = useState(!isExpanded);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const previousChatLength = useRef(externalChatHistory?.length || 0);
  const theme = useTheme();
  
  // Sync chat history from props
  useEffect(() => {
    if (externalChatHistory && externalChatHistory.length > 0) {
      setChatHistory(externalChatHistory);
    }
  }, [externalChatHistory]);
  
  // Sync expanded state with prop - with debounce to prevent immediate collapse
  useEffect(() => {
    let debounceTimeout: NodeJS.Timeout | null = null;
    
    // Only respond to external state changes when not in the middle of an animation
    if (isExpanded !== expanded && !minimizing) {
      setClickPrevented(true);
      
      if (isExpanded) {
        // Going from collapsed to expanded
        setShowCollapsed(false);
        debounceTimeout = setTimeout(() => {
          setExpanded(true);
          setClickPrevented(false);
        }, 100);
      } else {
        // Going from expanded to collapsed
        setMinimizing(true);
        debounceTimeout = setTimeout(() => {
          setExpanded(false);
          // Only show collapsed after all animations are done
          setTimeout(() => {
            setShowCollapsed(true);
            setMinimizing(false);
            setClickPrevented(false);
          }, 350);
        }, 300);
      }
    }
    
    return () => {
      if (debounceTimeout) {
        clearTimeout(debounceTimeout);
      }
    };
  }, [isExpanded, minimizing, expanded]);

  // Check for new chat messages from the server
  useEffect(() => {
    // If we have more chat messages than before, clear pending messages
    if (chatHistory.length > previousChatLength.current) {
      setPendingMessages([]);
    }
    previousChatLength.current = chatHistory.length;
  }, [chatHistory]);

  // Scroll to bottom of chat when new messages come in or when expanding
  useEffect(() => {
    if (messagesEndRef.current) {
      // Use a slight delay to ensure animations are complete
      const timer = setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [chatHistory, expanded, pendingMessages, isFullscreen]);

  // Automatically expand chat when sending first question
  useEffect(() => {
    if (pendingMessages.length > 0 && !expanded) {
      setExpanded(true);
      onToggle();
    }
  }, [pendingMessages, expanded, onToggle]);

  // Handle window resize smoothly
  useEffect(() => {
    const handleResize = () => {
      // Force a re-render to ensure proper sizing
      setIsFullscreen(prev => {
        if (prev) {
          // Trigger a small state change to force smooth animation
          setTimeout(() => {
            document.body.style.setProperty('--chat-resize', 'complete');
          }, 10);
        }
        return prev;
      });
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  // Prevent the issue where chat opens and immediately closes
  const handleCollapsedContainerClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    
    // Prevent multiple clicks in quick succession
    if (clickPrevented) return;
    
    // Set clickPrevented to true temporarily to debounce
    setClickPrevented(true);
    
    // Hide collapsed container immediately
    setShowCollapsed(false);
    
    // Set expanded directly, don't use toggle handler to avoid animation race conditions
    setTimeout(() => {
      setExpanded(true);
      onToggle();
      setTimeout(() => setClickPrevented(false), 100);
    }, 50);
  };
  
  const handleToggle = (e?: React.MouseEvent) => {
    // Prevent event propagation if this is from a click event
    if (e) {
      e.stopPropagation();
      e.preventDefault();
    }
    
    // Prevent multiple clicks in quick succession
    if (clickPrevented) return;
    
    // Prevent interaction during transitions
    setClickPrevented(true);
    
    if (expanded) {
      // Start minimize animation only
      setMinimizing(true);
      
      // After animation completes, actually collapse and notify parent
      setTimeout(() => {
        setExpanded(false);
        onToggle();
        
        // Wait for collapse animation to fully complete before showing collapsed button
        setTimeout(() => {
          setShowCollapsed(true);
          setMinimizing(false);
          setClickPrevented(false);
        }, 350); // Slightly longer than the Collapse animation
      }, 300); // Wait for our custom animation to complete
    } else {
      setShowCollapsed(false); // Immediately hide the collapsed container
      setExpanded(true);
      onToggle();
      setClickPrevented(false);
    }
  };

  const handleFullscreenToggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    
    // Add a CSS class to body to help with smoother transitions
    if (!isFullscreen) {
      document.body.classList.add('chat-fullscreen-active');
    } else {
      document.body.classList.remove('chat-fullscreen-active');
    }
    
    setIsFullscreen(!isFullscreen);
    
    // Scroll to bottom after transition completes
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, 300);
  };

  const handleSendQuestion = async () => {
    if (!userQuestion.trim() || loading) return;
    
    const currentQuestion = userQuestion.trim();
    setUserQuestion('');
    setLoading(true);
    
    // Immediately add the question to pending messages
    setPendingMessages([...pendingMessages, { 
      question: currentQuestion, 
      loading: true 
    }]);
    
    try {
      const response = await fetch(`${BACKEND_URL}/api/followup`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: currentQuestion,
          client_id: clientId,
        }),
      });
      
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.message || 'Failed to send question');
      }
      
      // The response will be handled through WebSocket updates to chatHistory
      // Don't clear pending messages here - they'll be cleared when the chat history updates
      
    } catch (error) {
      console.error('Error sending follow-up question:', error);
      // Keep the question in pending but mark as error
      setPendingMessages(prev => {
        const updated = [...prev];
        const lastIndex = updated.length - 1;
        if (lastIndex >= 0) {
          updated[lastIndex] = { 
            ...updated[lastIndex], 
            loading: false,
            error: true
          };
        }
        return updated;
      });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendQuestion();
    }
  };

  const renderMessages = () => {
    return (
      <>
        {chatHistory.length === 0 && pendingMessages.length === 0 ? (
          <Box 
            sx={{ 
              textAlign: 'center', 
              py: 4, 
              color: 'text.secondary',
              fontStyle: 'italic'
            }}
          >
            <Typography variant="body2">
              Ask a follow-up question about the research report
            </Typography>
          </Box>
        ) : (
          <>
            {/* Render chat history */}
            {chatHistory.map((message, index) => (
              <React.Fragment key={`history-${index}`}>
                <MessageBubble isUser={true}>
                  <Typography variant="body2">{message.question}</Typography>
                </MessageBubble>
                
                <MessageBubble isUser={false}>
                  <EnhancedMarkdown content={message.answer} sources={reportSources} />
                </MessageBubble>
              </React.Fragment>
            ))}
            
            {/* Render pending messages */}
            {pendingMessages.map((message, index) => (
              <React.Fragment key={`pending-${index}`}>
                <MessageBubble isUser={true}>
                  <Typography variant="body2">{message.question}</Typography>
                </MessageBubble>
                
                {message.loading && (
                  <PendingMessageBubble isUser={false}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <CircularProgress size={16} />
                      <Typography variant="body2">Generating response...</Typography>
                    </Box>
                  </PendingMessageBubble>
                )}
                
                {message.error && (
                  <MessageBubble isUser={false}>
                    <Typography variant="body2" color="error">
                      Error: Failed to generate response. Please try again.
                    </Typography>
                  </MessageBubble>
                )}
              </React.Fragment>
            ))}
          </>
        )}
        <div ref={messagesEndRef} />
      </>
    );
  };

  return (
    <>
      <GlobalStyles />
      {!expanded && showCollapsed && (
        <Fade in={!expanded && showCollapsed} timeout={300} mountOnEnter unmountOnExit>
          <CollapsedContainer 
            onClick={handleCollapsedContainerClick} 
            aria-label="Open chat"
            sx={{ pointerEvents: clickPrevented ? 'none' : 'auto' }}
          >
            <Avatar 
              sx={{ 
                mr: 1, 
                bgcolor: theme.palette.primary.main,
                transition: 'all 0.2s ease',
              }}
            >
              <QuestionAnswerIcon fontSize="small" />
            </Avatar>
            <Typography variant="subtitle2">Ask follow-up questions</Typography>
          </CollapsedContainer>
        </Fade>
      )}
      
      <Collapse 
        in={expanded} 
        timeout={{
          enter: 300,
          exit: minimizing ? 200 : 300 // Faster exit when minimizing to prevent overlap
        }}
        mountOnEnter 
        unmountOnExit
        addEndListener={(node, done) => {
          // Add a custom end listener to ensure proper sequencing
          node.addEventListener('transitionend', (event) => {
            if (event.target === node && !expanded) {
              // After Collapse transition completely ends and we're closing
              // Nothing more to do here, the setTimeout in handleToggle will handle showing collapsed
              done();
            } else {
              done();
            }
          }, { once: true }); // Use once:true so it automatically removes after firing
        }}
      >
        <ChatContainer 
          isFullscreen={isFullscreen}
          sx={{
            transform: minimizing ? 'translateY(20px) scale(0.95)' : 'translateY(0) scale(1)',
            opacity: minimizing ? 0 : 1,
            pointerEvents: clickPrevented ? 'none' : 'auto',
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            visibility: minimizing && !expanded ? 'hidden' : 'visible', // Hide completely at end of animation
          }}
        >
          <ChatHeader>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <QuestionAnswerIcon sx={{ mr: 1 }} />
              <Typography variant="subtitle1">
                Ask follow-up questions
              </Typography>
            </Box>
            <Box>
              <IconButton 
                size="small" 
                onClick={handleFullscreenToggle}
                sx={{ mr: 1 }}
                aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
                disabled={clickPrevented}
              >
                {isFullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />}
              </IconButton>
              <IconButton 
                size="small" 
                onClick={(e) => handleToggle(e)}
                aria-label="Close chat"
                disabled={clickPrevented}
              >
                <CloseIcon />
              </IconButton>
            </Box>
          </ChatHeader>
          
          <ChatMessages>
            {renderMessages()}
          </ChatMessages>
          
          <ChatInput className={loading ? 'disabled' : ''}>
            <TextField
              fullWidth
              variant="outlined"
              size="small"
              placeholder="Ask a follow-up question..."
              value={userQuestion}
              onChange={(e) => setUserQuestion(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={loading}
              multiline
              maxRows={3}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: '20px',
                  transition: 'all 0.2s ease',
                  backgroundColor: theme.palette.mode === 'dark' 
                    ? 'rgba(255, 255, 255, 0.05)' 
                    : 'rgba(0, 0, 0, 0.04)',
                  '&:hover': {
                    backgroundColor: theme.palette.mode === 'dark' 
                      ? 'rgba(255, 255, 255, 0.1)' 
                      : 'rgba(0, 0, 0, 0.06)',
                  },
                }
              }}
            />
            <IconButton 
              color="primary" 
              onClick={handleSendQuestion} 
              disabled={!userQuestion.trim() || loading}
              sx={{
                transition: 'all 0.2s ease',
                opacity: !userQuestion.trim() || loading ? 0.5 : 1,
                transform: userQuestion.trim() && !loading ? 'scale(1.05)' : 'scale(1)',
              }}
            >
              <SendIcon />
            </IconButton>
          </ChatInput>
        </ChatContainer>
      </Collapse>
    </>
  );
};

export default FollowupChat; 
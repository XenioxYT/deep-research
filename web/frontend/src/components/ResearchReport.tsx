import { Paper, Box, Fade, Typography, Container, useTheme, useMediaQuery, Tooltip, Card, CardContent } from '@mui/material';
import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { styled } from '@mui/material/styles';
import { useMemo } from 'react';

interface Source {
  id: string;
  title: string;
  url: string;
  domain: string;
  score: number;
}

interface ResearchReportProps {
  content: string;
}

const StyledMarkdown = styled(Box)(({ theme }) => ({
  '& .markdown-body': {
    color: theme.palette.text.primary,
    backgroundColor: 'transparent',
    fontFamily: theme.typography.fontFamily,
    fontSize: '1.1rem',
    lineHeight: 1.7,
    maxWidth: '100%',
    margin: '0 auto',
    padding: theme.spacing(0, 2),
    
    '& h1': {
      color: theme.palette.primary.main,
      fontSize: '2.5rem',
      fontWeight: 500,
      marginBottom: theme.spacing(4),
      borderBottom: `1px solid ${theme.palette.divider}`,
      paddingBottom: theme.spacing(1),
      fontFamily: '"Google Sans", sans-serif',
    },
    
    '& h2': {
      color: theme.palette.primary.light,
      fontSize: '2rem',
      fontWeight: 500,
      marginTop: theme.spacing(6),
      marginBottom: theme.spacing(3),
      borderBottom: `1px solid ${theme.palette.divider}`,
      paddingBottom: theme.spacing(1),
      fontFamily: '"Google Sans", sans-serif',
    },
    
    '& h3': {
      color: theme.palette.secondary.main,
      fontSize: '1.5rem',
      fontWeight: 500,
      marginTop: theme.spacing(4),
      marginBottom: theme.spacing(2),
      fontFamily: '"Google Sans", sans-serif',
    },
    
    '& p': {
      marginBottom: theme.spacing(2),
      fontSize: '1.1rem',
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
      padding: theme.spacing(1, 3),
      backgroundColor: 'rgba(187, 134, 252, 0.05)',
      borderRadius: `0 ${theme.shape.borderRadius}px ${theme.shape.borderRadius}px 0`,
      color: theme.palette.text.secondary,
    },
    
    '& table': {
      width: '100%',
      borderCollapse: 'separate',
      borderSpacing: 0,
      marginBottom: theme.spacing(3),
      '& th': {
        backgroundColor: 'rgba(187, 134, 252, 0.1)',
        color: theme.palette.primary.light,
        fontWeight: 500,
        textAlign: 'left',
        padding: theme.spacing(1.5),
        borderBottom: `2px solid ${theme.palette.divider}`,
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
      margin: theme.spacing(4, 0),
    },
    
    '& img': {
      maxWidth: '100%',
      height: 'auto',
      borderRadius: theme.shape.borderRadius,
      margin: theme.spacing(2, 0),
    },
  },
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
  minWidth: 300,
  maxWidth: 400,
  border: `1px solid ${theme.palette.divider}`,
  boxShadow: theme.shadows[10],
}));

const ResearchReport = ({ content }: ResearchReportProps) => {
  const theme = useTheme();
  const isLargeScreen = useMediaQuery(theme.breakpoints.up('md'));

  // Parse sources from the content
  const { processedContent, sources } = useMemo(() => {
    // Updated regex to match source entries more reliably
    const sourcesRegex = /\[(\d+)\]\s+(.+?)\n-\s*URL:\s*(.+?)\n-\s*Relevance Score:\s*([\d.]+)\n-\s*Domain:\s*(.+?)(?:\n|$)/g;
    const sources = new Map<string, Source>();
    let matches;
    let sourceSection = '';

    // Find all source entries at the end of the document
    let lastIndex = -1;
    while ((matches = sourcesRegex.exec(content)) !== null) {
      const [fullMatch, id, title, url, score, domain] = matches;
      const sourceKey = `[${id}]`;
      sources.set(sourceKey, {
        id: sourceKey,
        title: title.trim(),
        url: url.trim(),
        domain: domain.trim(),
        score: parseFloat(score),
      });
      sourceSection += fullMatch;
      lastIndex = matches.index;
    }

    // Remove the sources section from the content
    const processedContent = lastIndex > -1 ? content.slice(0, lastIndex).trim() : content;

    return { processedContent, sources };
  }, [content]);

  // Custom renderer for text to handle reference links
  const renderText = (text: React.ReactNode): React.ReactNode => {
    if (typeof text !== 'string') {
      return text;
    }

    // First process citations
    const parts = text.split(/(\[\d+\](?:\s*\[\d+\])*(?:[.,]?\s*|\.$|\)?|$))/g);
    const processedParts = parts.map((part, partIndex) => {
      const ref = part.match(/^\[\d+\]$/);
      if (ref) {
        const source = sources.get(ref[0]);
        if (source) {
          return (
            <SourceTooltip
              key={`${ref}-${partIndex}`}
              title={
                <SourceCard>
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>
                      {source.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" component="a" 
                      href={source.url} 
                      target="_blank"
                      rel="noopener noreferrer"
                      sx={{ 
                        display: 'block', 
                        mb: 1,
                        color: 'primary.main',
                        textDecoration: 'none',
                        '&:hover': {
                          textDecoration: 'underline',
                        },
                      }}
                    >
                      {source.url}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Relevance Score: {source.score.toFixed(2)}
                    </Typography>
                  </CardContent>
                </SourceCard>
              }
              placement="top"
              arrow
            >
              <Box
                component="span"
                sx={{
                  color: 'primary.main',
                  cursor: 'pointer',
                  '&:hover': {
                    textDecoration: 'underline',
                  },
                  display: 'inline',
                }}
              >
                {ref[0]}
              </Box>
            </SourceTooltip>
          );
        }
      }

      // Then process code blocks within non-citation text
      const codeBlocks: Array<{
        placeholder: string;
        content: string;
      }> = [];
      let currentIndex = 0;

      // Replace backtick content with placeholders and collect code blocks
      const textWithPlaceholders = part.replace(/`[^`]+`/g, (match) => {
        const placeholder = `__CODE_BLOCK_${currentIndex}__`;
        codeBlocks.push({
          placeholder,
          content: match.slice(1, -1) // Remove backticks
        });
        currentIndex++;
        return placeholder;
      });

      const elements: React.ReactNode[] = [];
      let remainingText = textWithPlaceholders;

      // Replace code block placeholders with actual components
      codeBlocks.forEach(({ placeholder, content }) => {
        if (remainingText.includes(placeholder)) {
          const [before, ...rest] = remainingText.split(placeholder);
          if (before) elements.push(before);
          
          elements.push(
            <Typography
              key={`code-${placeholder}`}
              component="code"
              sx={{
                backgroundColor: 'rgba(187, 134, 252, 0.1)',
                color: theme.palette.primary.light,
                padding: '2px 6px',
                borderRadius: 1,
                fontSize: '0.9em',
                fontFamily: 'Roboto Mono, monospace',
                display: 'inline',
              }}
            >
              {content}
            </Typography>
          );
          
          remainingText = rest.join(placeholder);
        }
      });

      if (remainingText) {
        elements.push(remainingText);
      }

      return elements.length > 0 ? (
        <React.Fragment key={`part-${partIndex}`}>{elements}</React.Fragment>
      ) : part;
    });

    return <>{processedParts}</>;
  };

  return (
    <Fade in={true} timeout={500}>
      <Paper
        elevation={3}
        sx={{
          backgroundColor: 'background.paper',
          borderRadius: 2,
          overflow: 'hidden',
        }}
      >
        <Container
          maxWidth="md"
          sx={{
            py: { xs: 4, md: 6 },
            px: { xs: 2, md: 4 },
          }}
        >
          <StyledMarkdown>
            <ReactMarkdown
              className="markdown-body"
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeKatex]}
              components={{
                p: ({ children }) => (
                  <Typography variant="body1" paragraph>
                    {renderText(children)}
                  </Typography>
                ),
                li: ({ children }) => {
                  // Helper function to extract text content from React nodes
                  const extractText = (nodes: React.ReactNode): string => {
                    if (typeof nodes === 'string') return nodes;
                    if (!nodes) return '';
                    
                    if (Array.isArray(nodes)) {
                      return nodes.map(extractText).join('');
                    }
                    
                    if (React.isValidElement(nodes)) {
                      const element = nodes as React.ReactElement;
                      return extractText(element.props?.children || '');
                    }
                    
                    return '';
                  };

                  // Get the full text content including any nested elements
                  const fullText = extractText(children);

                  return (
                    <Typography 
                      component="li"
                      sx={{
                        '& > *': { display: 'inline' },
                        '& strong, & em': { display: 'inline' },
                      }}
                    >
                      {renderText(fullText)}
                    </Typography>
                  );
                },
                strong: ({ children }) => (
                  <Typography
                    component="strong"
                    sx={{ 
                      fontWeight: 'bold',
                      display: 'inline',
                    }}
                  >
                    {renderText(children)}
                  </Typography>
                ),
                em: ({ children }) => (
                  <Typography
                    component="em"
                    sx={{ 
                      fontStyle: 'italic',
                      display: 'inline',
                    }}
                  >
                    {renderText(children)}
                  </Typography>
                ),
                a: ({ children, href }) => (
                  <Typography
                    component="a"
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    sx={{
                      color: 'primary.main',
                      textDecoration: 'none',
                      borderBottom: '1px solid',
                      borderColor: 'primary.main',
                      display: 'inline',
                      '&:hover': {
                        color: 'primary.light',
                        borderColor: 'primary.light',
                      },
                    }}
                  >
                    {renderText(children)}
                  </Typography>
                ),
                code: ({ node, inline, className, children, ...props }: {
                  node?: any;
                  inline?: boolean;
                  className?: string;
                  children: React.ReactNode;
                  [key: string]: any;
                }) => {
                  const match = /language-(\w+)/.exec(className || '');
                  const isLatex = match && match[1] === 'latex';
                  
                  if (inline) {
                    return (
                      <Typography
                        component="code"
                        sx={{
                          backgroundColor: 'rgba(187, 134, 252, 0.1)',
                          color: theme.palette.primary.light,
                          padding: '2px 6px',
                          borderRadius: 1,
                          fontSize: '0.9em',
                          fontFamily: 'Roboto Mono, monospace',
                          display: 'inline',
                        }}
                        {...props}
                      >
                        {children}
                      </Typography>
                    );
                  }

                  return (
                    <Box
                      component="pre"
                      sx={{
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
                      }}
                    >
                      <code className={className} {...props}>
                        {children}
                      </code>
                    </Box>
                  );
                },
              }}
            >
              {processedContent}
            </ReactMarkdown>
          </StyledMarkdown>
        </Container>
      </Paper>
    </Fade>
  );
};

export default ResearchReport; 
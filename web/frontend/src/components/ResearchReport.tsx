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
      margin: theme.spacing(4, 0),
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
    
    '& .katex-html': {
      whiteSpace: 'nowrap',
    }
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

const SourceBadge = styled(Box)(({ theme }) => ({
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

const SourceBadgeWrapper = ({ source, children }: { source: Source, children: React.ReactNode }) => {
  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    if (source.url) {
      window.open(source.url, '_blank', 'noopener,noreferrer');
    }
  };

  return (
    <SourceTooltip
      title={
        <SourceCard>
          <CardContent>
            <Typography variant="subtitle1" gutterBottom>
              {source.title}
            </Typography>
            {source.domain && (
              <Typography variant="body2" color="text.secondary">
                Domain: {source.domain}
              </Typography>
            )}
            <Typography variant="body2" color="text.secondary">
              Relevance Score: {source.score.toFixed(2)}
            </Typography>
          </CardContent>
        </SourceCard>
      }
      placement="top"
      arrow
    >
      <SourceBadge component="span" onClick={handleClick}>
        {children}
      </SourceBadge>
    </SourceTooltip>
  );
};

const ResearchReport = ({ content }: ResearchReportProps) => {
  const theme = useTheme();

  // Parse sources from the content
  const { processedContent, sources } = useMemo(() => {
    const sources = new Map<string, Source>();
    
    const sourcesSectionMatch = content.match(/(?:## Sources Used|## Sources|## References)\s*([\s\S]+)$/);
    if (!sourcesSectionMatch) {
      return { processedContent: content, sources };
    }

    const sourcesSection = sourcesSectionMatch[1];
    const mainContent = content.slice(0, sourcesSectionMatch.index).trim();
    
    // Split by numbered entries
    const sourceEntries = sourcesSection.split(/(?=\[\d+\])/);
    sourceEntries.forEach(entry => {
      const idMatch = entry.match(/^\[(\d+)\]/);
      if (!idMatch) return;

      const id = idMatch[1];
      const sourceKey = `[${id}]`;

      // Extract URL
      const urlMatch = entry.match(/URL:\s*(https?:\/\/[^\s\n]+)/i);
      const url = urlMatch ? urlMatch[1].trim() : '';

      // Extract relevance score
      const scoreMatch = entry.match(/Relevance Score:\s*([\d.]+)/i);
      const score = scoreMatch ? parseFloat(scoreMatch[1]) : 1.0;

      // Extract domain
      const domainMatch = entry.match(/Domain:\s*([^\s\n]+)/i);
      const domain = domainMatch ? domainMatch[1].trim() : (url ? new URL(url).hostname : '');

      // Get title by taking the first line after the ID, before any metadata
      let title = entry
        .split('\n')[0]         // Take first line
        .replace(/^\[\d+\]\s*/, '') // Remove ID
        .trim();

      if (title || url) {
        sources.set(sourceKey, {
          id: sourceKey,
          title,
          url,
          domain,
          score,
        });
      }
    });

    return { processedContent: mainContent, sources };
  }, [content]);

  // Fix JSX namespace issues
  type RenderSourceTooltipFn = (sourceKey: string) => React.ReactElement | null;

  const renderSourceTooltip = (sourceKey: string): React.ReactElement | null => {
    const source = sources.get(sourceKey);
    if (!source) return null;

    return (
      <SourceTooltip
        title={
          <SourceCard>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                {source.title}
              </Typography>
              {source.url && (
                <Typography 
                  variant="body2" 
                  color="text.secondary" 
                  component="div"
                  sx={{ mb: 1 }}
                >
                  <Box
                    component="a"
                    href={source.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    sx={{
                      color: 'primary.main',
                      textDecoration: 'none',
                      '&:hover': {
                        textDecoration: 'underline',
                      },
                    }}
                  >
                    {source.url}
                  </Box>
                </Typography>
              )}
              {source.domain && (
                <Typography variant="body2" color="text.secondary">
                  Domain: {source.domain}
                </Typography>
              )}
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
          {sourceKey}
        </Box>
      </SourceTooltip>
    );
  };

  const processChildren = (children: any, renderTooltip: RenderSourceTooltipFn) => {
    return React.Children.map(children, (child) => {
      if (typeof child === 'string') {
        const parts = child.split(/(\[\d+(?:\s*,\s*\d+)*\])/g);
        return parts.map((part, i) => {
          const sourceMatch = part.match(/\[(\d+(?:\s*,\s*\d+)*)\]/);
          if (sourceMatch) {
            const numbers = sourceMatch[1].split(/\s*,\s*/).map(num => num.trim());
            if (numbers.length === 1) {
              const source = sources.get(`[${numbers[0]}]`);
              if (!source) return numbers[0];
              
              return (
                <SourceBadgeWrapper key={i} source={source}>
                  {numbers[0]}
                </SourceBadgeWrapper>
              );
            }
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
                  const source = sources.get(`[${num}]`);
                  if (!source) return num;
                  
                  return (
                    <React.Fragment key={num}>
                      {index > 0 && ', '}
                      <SourceBadgeWrapper source={source}>
                        {num}
                      </SourceBadgeWrapper>
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

  const processParagraphChildren = (children: any, renderTooltip: RenderSourceTooltipFn) => {
    return <Typography variant="body1" paragraph>{processChildren(children, renderTooltip)}</Typography>;
  };

  const processTableCellChildren = (children: any, renderTooltip: RenderSourceTooltipFn) => {
    return <td>{processChildren(children, renderTooltip)}</td>;
  };

  const processListItemChildren = (children: any, renderTooltip: RenderSourceTooltipFn) => {
    return <li>{processChildren(children, renderTooltip)}</li>;
  };

  const components = useMemo(() => ({
    p: ({ children }: any) => processParagraphChildren(children, renderSourceTooltip),
    strong: ({ children }: any) => (
      <Typography component="strong" sx={{ fontWeight: 'bold', display: 'inline' }}>
        {children}
      </Typography>
    ),
    td: ({ children }: any) => processTableCellChildren(children, renderSourceTooltip),
    li: ({ children }: any) => processListItemChildren(children, renderSourceTooltip),
  }), [sources]);

  return (
    <Fade in={true} timeout={500}>
      <Paper
        elevation={3}
        sx={{
          backgroundColor: theme.palette.background.paper,
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
              remarkPlugins={[
                remarkGfm,
                [remarkMath, { inlineMath: [], displayMath: [['$$', '$$']] }]
              ]}
              rehypePlugins={[rehypeKatex]}
              components={components}
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
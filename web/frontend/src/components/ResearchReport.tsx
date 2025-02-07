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

  // Parse sources from the content
  const { processedContent, sources } = useMemo(() => {
    const sources = new Map<string, Source>();
    
    const sourcesSectionMatch = content.match(/(?:## Sources Used|## Sources|## References)\n\n([\s\S]+)$/);
    if (!sourcesSectionMatch) {
      return { processedContent: content, sources };
    }

    const sourcesSection = sourcesSectionMatch[1];
    const mainContent = content.slice(0, sourcesSectionMatch.index).trim();
    
    const sourceEntries = sourcesSection.split('\n\n');
    sourceEntries.forEach(entry => {
      const idMatch = entry.match(/^\[(\d+)\]/);
      if (!idMatch) return;

      const id = idMatch[1];
      const sourceKey = `[${id}]`;

      const urlMatch = entry.match(/URL:\s*(.+?)(?:\n|$)/);
      const url = urlMatch ? urlMatch[1].trim() : '';

      const titleMatch = entry.match(/^\[\d+\]\s+(.+?)(?:\n|$)/);
      const title = titleMatch ? titleMatch[1].trim() : '';

      const domainMatch = entry.match(/Domain:\s*(.+?)(?:\n|$)/);
      const domain = domainMatch ? domainMatch[1].trim() : new URL(url).hostname;

      const scoreMatch = entry.match(/(?:Relevance )?Score:\s*([\d.]+)/);
      const score = scoreMatch ? parseFloat(scoreMatch[1]) : 1.0;

      sources.set(sourceKey, {
        id: sourceKey,
        title,
        url,
        domain,
        score,
      });
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
          {sourceKey}
        </Box>
      </SourceTooltip>
    );
  };

  const processChildren = (children: any, renderTooltip: RenderSourceTooltipFn) => {
    return React.Children.map(children, (child) => {
      if (typeof child === 'string') {
        const parts = child.split(/(\[\d+\])/g);
        return parts.map((part, i) => {
          const sourceMatch = part.match(/\[(\d+)\]/);
          if (sourceMatch) {
            const element = renderTooltip(part);
            return element ? <React.Fragment key={i}>{element}</React.Fragment> : part;
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
              remarkPlugins={[remarkGfm, remarkMath]}
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
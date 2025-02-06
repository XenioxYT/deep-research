import { Paper, Box, Fade } from '@mui/material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ResearchReportProps {
  content: string;
}

const ResearchReport = ({ content }: ResearchReportProps) => {
  return (
    <Fade in={true}>
      <Paper
        elevation={3}
        sx={{
          p: 4,
          backgroundColor: 'background.paper',
          borderRadius: 2,
          '& img': {
            maxWidth: '100%',
            height: 'auto',
          },
        }}
      >
        <Box
          sx={{
            '& .markdown-body': {
              color: 'text.primary',
              backgroundColor: 'transparent',
              fontFamily: 'inherit',
              fontSize: '1rem',
              lineHeight: 1.7,
              '& h1': {
                color: 'primary.main',
                borderBottom: '1px solid',
                borderColor: 'divider',
                pb: 1,
                mb: 3,
              },
              '& h2': {
                color: 'primary.light',
                borderBottom: '1px solid',
                borderColor: 'divider',
                pb: 1,
                mb: 2,
                mt: 4,
              },
              '& h3': {
                color: 'secondary.main',
                mb: 2,
                mt: 3,
              },
              '& a': {
                color: 'primary.main',
                textDecoration: 'none',
                '&:hover': {
                  textDecoration: 'underline',
                },
              },
              '& code': {
                backgroundColor: 'rgba(187, 134, 252, 0.1)',
                color: 'primary.light',
                padding: '2px 4px',
                borderRadius: 1,
                fontSize: '0.9em',
              },
              '& pre': {
                backgroundColor: 'rgba(187, 134, 252, 0.05)',
                padding: 2,
                borderRadius: 1,
                overflow: 'auto',
                '& code': {
                  backgroundColor: 'transparent',
                  padding: 0,
                },
              },
              '& blockquote': {
                borderLeft: '4px solid',
                borderColor: 'primary.main',
                ml: 0,
                pl: 2,
                color: 'text.secondary',
              },
              '& table': {
                borderCollapse: 'collapse',
                width: '100%',
                mb: 3,
                '& th': {
                  backgroundColor: 'rgba(187, 134, 252, 0.1)',
                  borderBottom: '2px solid',
                  borderColor: 'divider',
                  padding: 1,
                  textAlign: 'left',
                },
                '& td': {
                  borderBottom: '1px solid',
                  borderColor: 'divider',
                  padding: 1,
                },
              },
              '& ul, & ol': {
                pl: 3,
                mb: 2,
              },
              '& li': {
                mb: 1,
              },
              '& hr': {
                border: 'none',
                borderTop: '1px solid',
                borderColor: 'divider',
                my: 3,
              },
            },
          }}
        >
          <ReactMarkdown
            className="markdown-body"
            remarkPlugins={[remarkGfm]}
          >
            {content}
          </ReactMarkdown>
        </Box>
      </Paper>
    </Fade>
  );
};

export default ResearchReport; 
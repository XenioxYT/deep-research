import { useState, useEffect } from 'react';
import { Paper, InputBase, IconButton, Box, Typography, CircularProgress, Slide } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import { keyframes } from '@mui/system';

const gradientAnimation = keyframes`
  0% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
  100% {
    background-position: 0% 50%;
  }
`;

interface SearchBoxProps {
  onSearch: (query: string) => void;
  disabled?: boolean;
  isResearching: boolean;
  currentQuery: string | null;
  isLoading: boolean;
}

const SearchBox = ({ onSearch, disabled = false, isResearching, currentQuery, isLoading }: SearchBoxProps) => {
  const [query, setQuery] = useState('');

  // Update local query when currentQuery changes
  useEffect(() => {
    if (currentQuery) {
      setQuery(currentQuery);
    }
  }, [currentQuery]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !disabled) {
      onSearch(query.trim());
    }
  };

  return (
    <Box 
      sx={{ 
        textAlign: 'center',
        transition: 'all 0.3s ease-in-out',
        mt: isResearching ? 2 : 8,
        mb: isResearching ? 2 : 4,
        width: '100%',
        opacity: isLoading ? 0.7 : 1,
      }}
    >
      <Slide direction="down" in={!isResearching} mountOnEnter unmountOnExit>
        <Typography
          variant="h3"
          component="h1"
          sx={{
            mb: 4,
            fontWeight: 500,
            fontFamily: '"Google Sans", sans-serif',
            background: 'linear-gradient(90deg, #8B5CF6, #EC4899, #3B82F6, #8B5CF6)',
            backgroundSize: '300% 100%',
            animation: `${gradientAnimation} 8s ease infinite`,
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}
        >
          Deep Research
        </Typography>
      </Slide>
      <Paper
        component="form"
        onSubmit={handleSubmit}
        elevation={3}
        sx={{
          p: '4px',
          display: 'flex',
          alignItems: 'center',
          width: isResearching ? '100%' : '600px',
          mx: 'auto',
          transition: 'all 0.3s ease-in-out',
          '&:hover': {
            transform: (disabled || isLoading) ? 'none' : 'translateY(-2px)',
            boxShadow: (theme) => (disabled || isLoading) ? theme.shadows[3] : theme.shadows[6],
          },
          opacity: isLoading ? 0.7 : 1,
          pointerEvents: isLoading ? 'none' : 'auto',
        }}
      >
        <InputBase
          sx={{
            ml: 2,
            flex: 1,
            fontSize: '1.1rem',
            '& input': {
              padding: '12px 0',
            },
          }}
          placeholder={isResearching ? currentQuery || '' : "What would you like to research?"}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={disabled || isLoading}
        />
        <IconButton
          type="submit"
          sx={{
            p: '12px',
            color: 'primary.main',
            '&:hover': {
              background: 'rgba(187, 134, 252, 0.1)',
            },
          }}
          disabled={disabled || isLoading}
        >
          {isResearching || isLoading ? <CircularProgress size={24} /> : <SearchIcon />}
        </IconButton>
      </Paper>
    </Box>
  );
};

export default SearchBox; 
import { Box, TextField, Button, Typography, CircularProgress } from '@mui/material';
import { useState } from 'react';

interface SearchBoxProps {
  onSearch: (query: string) => void;
  disabled: boolean;
  isResearching: boolean;
  currentQuery: string | null;
}

const SearchBox = ({ onSearch, disabled, isResearching, currentQuery }: SearchBoxProps) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
    }
  };

  return (
    <Box
      component="form"
      onSubmit={handleSubmit}
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 4,
      }}
    >
      <Typography
        variant="h2"
        sx={{
          fontFamily: '"Google Sans", sans-serif',
          fontWeight: 500,
          position: 'relative',
          color: '#BB86FC',
          textShadow: '0 0 20px rgba(187, 134, 252, 0.3)',
          mb: 2,
          '&::before': {
            content: 'attr(data-text)',
            position: 'absolute',
            left: 0,
            top: 0,
            width: '100%',
            height: '100%',
            background: 'linear-gradient(90deg, #BB86FC, #4285f4, #BB86FC)',
            backgroundSize: '200% 100%',
            animation: 'shimmer 6s ease-in-out infinite',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          },
          '@keyframes shimmer': {
            '0%': {
              backgroundPosition: '100% 0',
            },
            '100%': {
              backgroundPosition: '-100% 0',
            },
          },
        }}
        data-text="Deep Research"
      >
        Deep Research
      </Typography>

      <Box
        sx={{
          display: 'flex',
          gap: 2,
          width: '100%',
          maxWidth: 800,
        }}
      >
        <Box
          sx={{
            position: 'relative',
            flexGrow: 1,
            '&:hover::before, &:focus-within::before': {
              content: '""',
              position: 'absolute',
              top: -2,
              left: -2,
              right: -2,
              bottom: -2,
              background: 'linear-gradient(-45deg, #BB86FC, #4285f4)',
              backgroundSize: '200% 200%',
              animation: 'gradient 5s ease infinite',
              borderRadius: '14px',
              zIndex: 0,
              opacity: 0.5,
              filter: 'blur(8px)',
              transition: 'opacity 0.3s ease-in-out',
            },
            '@keyframes gradient': {
              '0%': { backgroundPosition: '0% 50%' },
              '50%': { backgroundPosition: '100% 50%' },
              '100%': { backgroundPosition: '0% 50%' },
            },
          }}
        >
          <TextField
            fullWidth
            placeholder="Enter your research query..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={disabled}
            sx={{
              position: 'relative',
              zIndex: 1,
              '& .MuiOutlinedInput-root': {
                backgroundColor: 'background.paper',
                transition: 'all 0.3s ease-in-out',
                '& .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'rgba(255, 255, 255, 0.1)',
                  transition: 'border-color 0.3s ease-in-out',
                },
                '&:hover .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'rgba(255, 255, 255, 0.2)',
                },
                '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                  borderColor: '#BB86FC',
                },
              },
            }}
          />
        </Box>

        <Button
          type="submit"
          variant="contained"
          disabled={disabled || !query.trim()}
          sx={{
            minWidth: 120,
            backgroundColor: 'primary.main',
            color: 'background.paper',
            position: 'relative',
            transition: 'all 0.3s ease-in-out',
            '&:hover': {
              backgroundColor: 'primary.dark',
              transform: 'translateY(-1px)',
            },
            '&.Mui-disabled': {
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
            },
            '&::before': {
              content: '""',
              position: 'absolute',
              top: -4,
              left: -4,
              right: -4,
              bottom: -4,
              background: 'linear-gradient(90deg, transparent, rgba(187, 134, 252, 0.4), transparent)',
              backgroundSize: '200% 100%',
              borderRadius: 'inherit',
              opacity: 0,
              filter: 'blur(8px)',
              animation: query.trim() ? 'buttonPulse 1.5s ease forwards' : 'none',
            },
            '@keyframes buttonPulse': {
              '0%': {
                opacity: 0,
                backgroundPosition: '200% 0',
              },
              '50%': {
                opacity: 0.7,
              },
              '100%': {
                opacity: 0,
                backgroundPosition: '-100% 0',
              },
            },
          }}
        >
          {isResearching ? (
            <CircularProgress size={24} color="inherit" />
          ) : (
            'Research'
          )}
        </Button>
      </Box>

      {currentQuery && (
        <Typography
          variant="body2"
          sx={{
            color: 'text.secondary',
            mt: -2,
          }}
        >
          Current query: {currentQuery}
        </Typography>
      )}
    </Box>
  );
};

export default SearchBox; 
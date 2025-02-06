import { useState } from 'react';
import { Paper, InputBase, IconButton, Box, Typography, CircularProgress } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';

interface SearchBoxProps {
  onSearch: (query: string) => void;
  disabled?: boolean;
}

const SearchBox = ({ onSearch, disabled = false }: SearchBoxProps) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !disabled) {
      onSearch(query.trim());
    }
  };

  return (
    <Box sx={{ textAlign: 'center', mt: 8, mb: 4 }}>
      <Typography
        variant="h3"
        component="h1"
        sx={{
          mb: 4,
          fontWeight: 500,
          background: 'linear-gradient(45deg, #BB86FC 30%, #03DAC6 90%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}
      >
        Deep Research
      </Typography>
      <Paper
        component="form"
        onSubmit={handleSubmit}
        elevation={3}
        sx={{
          p: '4px',
          display: 'flex',
          alignItems: 'center',
          maxWidth: 600,
          mx: 'auto',
          transition: 'all 0.3s ease-in-out',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: (theme) => theme.shadows[6],
          },
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
          placeholder="What would you like to research?"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={disabled}
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
          disabled={disabled}
        >
          {disabled ? <CircularProgress size={24} /> : <SearchIcon />}
        </IconButton>
      </Paper>
    </Box>
  );
};

export default SearchBox; 
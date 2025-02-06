import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#BB86FC',
      light: '#E0B8FF',
      dark: '#985EF7',
    },
    secondary: {
      main: '#03DAC6',
      light: '#66FFF8',
      dark: '#00A896',
    },
    error: {
      main: '#CF6679',
    },
    background: {
      default: '#121212',
      paper: '#1E1E1E',
    },
    text: {
      primary: '#FFFFFF',
      secondary: 'rgba(255, 255, 255, 0.7)',
    },
  },
  typography: {
    fontFamily: '"Google Sans", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontFamily: '"Google Sans", sans-serif',
      fontWeight: 500,
    },
    h2: {
      fontFamily: '"Google Sans", sans-serif',
      fontWeight: 500,
    },
    h3: {
      fontFamily: '"Google Sans", sans-serif',
      fontWeight: 500,
    },
    h4: {
      fontFamily: '"Google Sans", sans-serif',
      fontWeight: 500,
    },
    h5: {
      fontFamily: '"Google Sans", sans-serif',
      fontWeight: 500,
    },
    h6: {
      fontFamily: '"Google Sans", sans-serif',
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 24,
          padding: '8px 24px',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 12,
          },
        },
      },
    },
  },
});

export default theme; 
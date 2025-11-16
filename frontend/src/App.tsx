import { useEffect, useState } from "react";
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  CardMedia,
  createTheme,
  ThemeProvider,
} from "@mui/material";

const theme = createTheme({
  palette: {
    mode: "light",
    primary: { main: "#4285F4" },
    secondary: { main: "#EA4335" },
  },
  shape: {
    borderRadius: 16,
  },
});

export default function App() {
  const [files, setFiles] = useState([]);

  useEffect(() => {
    fetch("http://localhost:8000/api/plots")
      .then((res) => res.json())
      .then((data) => setFiles(data));
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <Container maxWidth="lg" style={{ paddingTop: 40, paddingBottom: 40 }}>
        <Typography variant="h4" gutterBottom fontWeight="bold">
          🎨 Color Perception Model — Visual Report
        </Typography>

        <Typography variant="subtitle1" color="text.secondary" gutterBottom>
          基于深度学习的色差预测模型 — 模型性能可视化（Google Material 风格）
        </Typography>

        <Grid container spacing={3} style={{ marginTop: 20 }}>
          {files.map((file) => (
            <Grid item xs={12} sm={6} md={6} key={file}>
              <Card elevation={3}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    {file.replace(".png", "").replaceAll("_", " ")}
                  </Typography>
                  <CardMedia
                    component="img"
                    height="300"
                    image={`http://localhost:8000/plots/${file}`}
                    style={{ objectFit: "contain", padding: 8 }}
                  />
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Container>
    </ThemeProvider>
  );
}

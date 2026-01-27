@echo off
echo 🍌 Starting Banana Ripeness Classifier Dashboard
echo ================================================
echo.

echo 📦 Checking if models directory exists...
if not exist "models\" (
    echo ⚠️  Creating models directory...
    mkdir models
    echo ⚠️  Please place your .pth model files in the models\ directory
)

echo.
echo 🐍 Starting Backend Server (FastAPI)...
start "Backend Server" cmd /k "cd backend && python app.py"

echo ⏳ Waiting for backend to start...
timeout /t 3 /nobreak > nul

echo.
echo ⚛️  Starting Frontend Server (Vite)...
start "Frontend Server" cmd /k "npm run dev"

echo.
echo ✅ Both servers are starting!
echo.
echo 📍 Backend:  http://localhost:8000
echo 📍 Frontend: http://localhost:3000
echo.
echo Close the server windows to stop the application
pause

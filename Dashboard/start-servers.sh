#!/bin/bash

echo "🍌 Starting Banana Ripeness Classifier Dashboard"
echo "================================================"

echo ""
echo "📦 Checking if models directory exists..."
if [ ! -d "models" ]; then
    echo "⚠️  Creating models directory..."
    mkdir models
    echo "⚠️  Please place your .pth model files in the models/ directory"
fi

echo ""
echo "🐍 Starting Backend Server (FastAPI)..."
cd backend
python app.py &
BACKEND_PID=$!
cd ..

echo "⏳ Waiting for backend to start..."
sleep 3

echo ""
echo "⚛️  Starting Frontend Server (Vite)..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Both servers are starting!"
echo ""
echo "📍 Backend:  http://localhost:8000"
echo "📍 Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"

trap "echo ''; echo '🛑 Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT

wait

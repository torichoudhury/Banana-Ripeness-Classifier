# Quick Start - Get Running in 5 Minutes

## Prerequisites Check

Do you have these installed?
- [ ] Node.js (check: `node --version`)
- [ ] Python 3.8+ (check: `python --version`)
- [ ] Your three `.pth` model files

## Step 1: Copy Your Models (30 seconds)

Copy your three PyTorch model files to the `models/` folder:

```bash
# Copy your .pth files here:
models/
├── banana_ripeness_efficientnet_pytorch.pth
├── banana_ripeness_resnet_pytorch.pth
└── banana_ripeness_mobilenet_pytorch.pth
```

## Step 2: Install Everything (2-3 minutes)

```bash
# Install frontend packages
npm install

# Install backend packages
cd backend
pip install -r requirements.txt
cd ..
```

## Step 3: Start It Up (30 seconds)

### Easy Way - One Command

**On Mac/Linux:**
```bash
chmod +x start-servers.sh
./start-servers.sh
```

**On Windows:**
```bash
start-servers.bat
```

### Manual Way - Two Terminals

**Terminal 1 (Backend):**
```bash
cd backend
python app.py
```

**Terminal 2 (Frontend):**
```bash
npm run dev
```

## Step 4: Use It!

1. Open browser: `http://localhost:3000`
2. Upload a banana image
3. Click "Classify Ripeness"
4. See results from all three models!

## That's It!

You should see:
- ✅ Backend running on port 8000
- ✅ Frontend running on port 3000
- ✅ Dashboard in your browser
- ✅ Three model predictions when you upload an image

## Common Issues

### "No models available"
- Check your `.pth` files are in the `models/` folder
- Check the file names match exactly

### "Port 8000 already in use"
- Stop any other applications using port 8000
- Or change the port in `backend/app.py`

### "Port 3000 already in use"
- Stop any other applications using port 3000
- Or change the port in `vite.config.js`

### Backend won't start
```bash
# Try installing dependencies again
cd backend
pip install -r requirements.txt
```

### Frontend won't start
```bash
# Try installing dependencies again
npm install
```

## What You'll See

### Dashboard Home
- 🍌 Big title with banana emoji
- 📤 Upload area (drag & drop)
- Clean, modern design

### After Upload
- 🖼️ Image preview
- 🔍 "Classify Ripeness" button
- 🔄 "Upload New Image" button

### After Prediction
- ⚡ EfficientNet-B0 results
- 🎯 ResNet50 results
- 📱 MobileNetV2 results
- 📊 Statistics comparing all models
- 📈 Confidence scores and probability bars

## Next Steps

- Read `README.md` for full documentation
- Check `PROJECT_OVERVIEW.md` to understand the architecture
- Customize the UI in `src/App.css`
- Deploy to production when ready

## Need Help?

1. Check the console output for error messages
2. Read `SETUP.md` for detailed troubleshooting
3. Verify all model files are in place
4. Make sure both servers are running

## Tips

- The dashboard works on mobile too!
- You can upload multiple images (one at a time)
- Compare which model performs best for your use case
- The confidence scores help you understand model certainty

Enjoy your banana classifier! 🍌🎉

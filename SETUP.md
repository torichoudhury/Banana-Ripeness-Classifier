# Quick Setup Guide

## Step 1: Copy Your Model Files

Copy your three trained PyTorch model files to the `models/` directory:

```
models/
├── banana_ripeness_efficientnet_pytorch.pth
├── banana_ripeness_resnet_pytorch.pth
└── banana_ripeness_mobilenet_pytorch.pth
```

These are the `.pth` files you got from training your models.

## Step 2: Install Dependencies

### Frontend Dependencies
```bash
npm install
```

### Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
cd ..
```

## Step 3: Start the Application

### Option 1: Use the Helper Script

**On Linux/Mac:**
```bash
chmod +x start-servers.sh
./start-servers.sh
```

**On Windows:**
```bash
start-servers.bat
```

### Option 2: Manual Start

**Terminal 1 - Backend:**
```bash
cd backend
python app.py
```

**Terminal 2 - Frontend:**
```bash
npm run dev
```

## Step 4: Access the Dashboard

Open your browser and go to: `http://localhost:3000`

## Testing the Dashboard

1. Click on the upload area or drag and drop a banana image
2. Click "Classify Ripeness"
3. View predictions from all three models
4. Analyze model statistics and confidence scores

## Troubleshooting

### "No models available" error
- Check that your `.pth` files are in the `models/` directory
- Verify the file names match exactly
- Check the backend console for loading errors

### Backend not starting
- Ensure Python 3.8+ is installed
- Install missing dependencies: `pip install -r backend/requirements.txt`
- Check port 8000 is not in use

### Frontend not starting
- Ensure Node.js 18+ is installed
- Run `npm install` again
- Check port 3000 is not in use

## GPU Support

The backend will automatically use CUDA/GPU if available. For CPU-only:
- The app will still work, just slower
- Consider reducing batch processing if needed

## Next Steps

- Upload different banana images to test
- Compare predictions across models
- Note which model performs best for your use case
- Check model agreement statistics

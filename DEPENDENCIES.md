# Project Dependencies

## Frontend Dependencies (Node.js)

### Core Dependencies
```json
{
  "react": "^18.2.0",           // UI library
  "react-dom": "^18.2.0",       // React DOM renderer
  "axios": "^1.6.2"             // HTTP client for API calls
}
```

### Development Dependencies
```json
{
  "@types/react": "^18.2.43",           // React TypeScript types
  "@types/react-dom": "^18.2.17",       // React DOM TypeScript types
  "@vitejs/plugin-react": "^4.2.1",     // Vite plugin for React
  "vite": "^5.0.8"                      // Build tool and dev server
}
```

**Install with:**
```bash
npm install
```

## Backend Dependencies (Python)

### Core Dependencies

```
fastapi==0.104.1
```
- Modern web framework for building APIs
- Automatic API documentation
- Type hints and validation
- Fast performance

```
uvicorn[standard]==0.24.0
```
- ASGI server for running FastAPI
- WebSocket support
- Production-ready
- Hot reload in development

```
python-multipart==0.0.6
```
- Required for file upload support
- Handles multipart/form-data
- Used for image uploads

```
torch>=2.0.0
```
- PyTorch deep learning framework
- Neural network computations
- Model loading and inference
- GPU acceleration support

```
torchvision>=0.15.0
```
- Computer vision library for PyTorch
- Pre-trained models (ResNet, MobileNet, EfficientNet)
- Image transformations
- Data augmentation utilities

```
Pillow>=10.0.0
```
- Python Imaging Library (PIL)
- Image loading and processing
- Format conversion
- Image manipulation

**Install with:**
```bash
cd backend
pip install -r requirements.txt
```

## Optional Dependencies

### For Production Deployment

```bash
pip install gunicorn
```
- Production WSGI/ASGI server
- Process management
- Worker spawning
- Better performance than uvicorn alone

### For Development

```bash
pip install python-dotenv
```
- Load environment variables from .env file
- Configuration management

```bash
pip install black flake8
```
- Code formatting (black)
- Code linting (flake8)
- Code quality tools

## System Requirements

### Minimum Requirements
- **Node.js**: v18.0.0 or higher
- **Python**: 3.8 or higher
- **RAM**: 4GB
- **Storage**: 2GB free space (for models and dependencies)

### Recommended Requirements
- **Node.js**: v20.0.0 or higher
- **Python**: 3.10 or higher
- **RAM**: 8GB or more
- **GPU**: NVIDIA GPU with CUDA support (optional)
- **Storage**: 5GB free space

## Platform-Specific Notes

### Windows
- May need Visual C++ Build Tools for some Python packages
- Use Command Prompt or PowerShell
- Run `start-servers.bat` to start both servers

### macOS
- May need Xcode Command Line Tools
- Use Terminal
- Run `./start-servers.sh` to start both servers

### Linux
- Usually works out of the box
- May need `python3-dev` package
- Run `./start-servers.sh` to start both servers

## GPU Support

### CUDA (NVIDIA GPUs)
```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### MPS (Apple Silicon)
PyTorch automatically detects and uses Apple's Metal Performance Shaders on M1/M2 Macs.

### CPU-Only
The default installation uses CPU. The application will work but inference will be slower.

## Version Compatibility

| Package | Minimum | Recommended |
|---------|---------|-------------|
| Node.js | 18.0 | 20.0+ |
| Python | 3.8 | 3.10+ |
| PyTorch | 2.0 | 2.1+ |
| FastAPI | 0.100 | 0.104+ |
| React | 18.0 | 18.2+ |

## Troubleshooting Dependencies

### Node.js Issues
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules
npm install
```

### Python Issues
```bash
# Upgrade pip
pip install --upgrade pip

# Install in virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```

### PyTorch Installation Issues
- Visit https://pytorch.org/get-started/locally/
- Select your OS, package manager, and CUDA version
- Use the generated install command

## Package Sizes (Approximate)

- **Frontend (node_modules)**: ~150 MB
- **Backend (Python packages)**: ~2 GB (mainly PyTorch)
- **Trained Models**: ~500 MB - 1 GB total
- **Total Project**: ~3-4 GB

## Security Updates

Keep dependencies updated:
```bash
# Frontend
npm update
npm audit fix

# Backend
pip install --upgrade -r backend/requirements.txt
```

## License Information

All dependencies use permissive open-source licenses:
- React: MIT
- Vite: MIT
- FastAPI: MIT
- PyTorch: BSD-style
- Pillow: HPND License

Check individual package licenses before commercial use.

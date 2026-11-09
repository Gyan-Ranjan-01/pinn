FROM python:3.10-slim

WORKDIR /app

# Upgrade pip and install lightweight CPU PyTorch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application source code
COPY . .

# Expose port (default 7860, or dynamically bound by Render/Railway via $PORT)
EXPOSE 7860

# Run FastAPI app with dynamic port binding
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-7860}"]

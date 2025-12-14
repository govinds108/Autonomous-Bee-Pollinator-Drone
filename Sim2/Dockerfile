FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgl1-mesa-dri \
    libglx-mesa0 \
    libglu1-mesa \
    mesa-utils \
    libglib2.0-0 \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxi6 \
    libxfixes3 \
    libxxf86vm1 \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/Sim2

COPY Sim2/requirements.txt /app/Sim2/requirements.txt
COPY Sim2 /app/Sim2

RUN python -m pip install --upgrade pip \
  && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.2+cpu torchvision==0.16.2+cpu \
  && python -m pip install --no-cache-dir -r Sim2/requirements.txt

CMD ["python", "Sim2/train_qlearn_follow.py", "--sim", "pybullet", "--no-gui"]

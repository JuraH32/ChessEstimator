# Step 1: Build the React app
FROM node:18 AS frontend
WORKDIR /chess-estimator-web-app
COPY chess-estimator-web-app/package.json chess-estimator-web-app/package-lock.json ./
RUN npm install
COPY chess-estimator-web-app/ .
RUN npm run build

# Step 2: Prepare the Python backend
FROM python:3.8-slim AS backend
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend .

# Final image with both frontend and backend
FROM nginx:stable AS final
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv supervisor && rm -rf /var/lib/apt/lists/*

# Copy Nginx and React setup
COPY --from=frontend /chess-estimator-web-app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy FastAPI setup
COPY --from=backend /app /app
WORKDIR /app

# Install Uvicorn
RUN python3 -m venv venv
RUN venv/bin/pip install --no-cache-dir -r requirements.txt

# Install Supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Build lc0 in maia folder
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y git ninja-build \
        build-essential python3 python3-pip \
        libeigen3-dev libopenblas-dev && \
        python3 -m venv /opt/venv && \
        /opt/venv/bin/pip install meson && \
        apt-get clean && rm -rf /var/lib/apt/lists/*

# Add virtual environment's bin directory to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Clone lc0
RUN git clone -b v0.31.2 --recurse-submodules https://github.com/LeelaChessZero/lc0.git /app/lc0
RUN sed -i "s/march=native/march=native/g" /app/lc0/meson.build && /app/lc0/build.sh -Dneon=false

## Copy the lc0 binary to app/maia
RUN cp /app/lc0/build/release/lc0 /app/maia
RUN rm -rf /app/lc0

# Expose necessary ports
EXPOSE 80 8000

# Start Supervisor
CMD ["supervisord", "-n"]

#!/bin/bash

# Tạo thư mục ssl
mkdir -p .streamlit/ssl

# Tạo SSL certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout .streamlit/ssl/key.pem \
    -out .streamlit/ssl/cert.pem \
    -subj "/C=VN/ST=HN/L=Hanoi/O=Local/OU=Dev/CN=localhost"

# Set permissions
chmod 600 .streamlit/ssl/key.pem
chmod 600 .streamlit/ssl/cert.pem

# Cài đặt dependencies nếu chưa có
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    sudo apt-get update
    sudo apt-get install -y ffmpeg
fi

# Chạy ứng dụng
streamlit run app.py 
#!/bin/bash

# Tạo thư mục ssl
mkdir -p .streamlit/ssl

# Tạo SSL certificate với SAN (Subject Alternative Names)
cat > .streamlit/ssl/openssl.conf << EOF
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
C = VN
ST = Hanoi
L = Hanoi
O = Local Development
OU = Development
CN = tr1nh.net

[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = tr1nh.net
DNS.2 = *.tr1nh.net
DNS.3 = localhost
EOF

# Tạo SSL certificate với config file
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout .streamlit/ssl/key.pem \
    -out .streamlit/ssl/cert.pem \
    -config .streamlit/ssl/openssl.conf \
    -extensions v3_req

# Set permissions
chmod 600 .streamlit/ssl/key.pem
chmod 600 .streamlit/ssl/cert.pem

# Cài đặt dependencies nếu chưa có
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    sudo apt-get update
    sudo apt-get install -y ffmpeg
fi

# Chạy ứng dụng với SSL
streamlit run app.py 
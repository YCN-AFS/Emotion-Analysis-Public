# Tạo thư mục ssl nếu chưa tồn tại
New-Item -ItemType Directory -Force -Path .streamlit/ssl

# Tạo self-signed certificate
$cert = New-SelfSignedCertificate -DnsName "localhost" -CertStoreLocation "Cert:\LocalMachine\My"

# Export certificate
$certPath = ".streamlit/ssl/cert.pem"
$keyPath = ".streamlit/ssl/key.pem"
$pwd = ConvertTo-SecureString -String "password" -Force -AsPlainText
Export-PfxCertificate -Cert $cert -FilePath "temp.pfx" -Password $pwd
openssl pkcs12 -in temp.pfx -out $certPath -nokeys -password pass:password
openssl pkcs12 -in temp.pfx -out $keyPath -nocerts -nodes -password pass:password

# Xóa file tạm
Remove-Item temp.pfx

# Chạy ứng dụng
streamlit run app.py 
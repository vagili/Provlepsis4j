# ---------- build stage ----------
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
# Prefer reproducible installs; fallback to install if no lock present
RUN npm ci || npm install
COPY . .
# API base is embedded at build time
ARG VITE_API=http://localhost:8080
ENV VITE_API=${VITE_API}
RUN npm run build

# ---------- runtime stage (no Node) ----------
FROM nginx:alpine
# SPA fallback so routing works
RUN printf 'server { \
  listen 80; \
  server_name _; \
  root /usr/share/nginx/html; \
  location / { try_files $uri /index.html; } \
}\n' > /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist/ /usr/share/nginx/html/

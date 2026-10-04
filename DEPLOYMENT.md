# Brain Tumour AI — Vercel + Render + Aiven deployment

The application is now separated into three responsibilities:

- `frontend/` — React + Vite application for Vercel.
- `backend/` — Flask + PyTorch API for Render.
- Aiven MySQL — persistent user/account database.

## 1. Aiven MySQL

Create or use an Aiven for MySQL service.

From Aiven Console > your MySQL service > Connection information, copy the Service URI. The URI contains the host, port, username, password, database and SSL mode.

Use that URI as Render's `DATABASE_URL`.

The backend creates the `users` table automatically on startup.

## 2. Render backend

Create a new Render Web Service from this repository.

Recommended settings:

- Branch: `main` after the migration is merged
- Root Directory: `backend`
- Runtime: Python
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn --workers 1 --threads 1 --timeout 180 app:app`
- Plan: Free

Environment variables:

- `DATABASE_URL` = Aiven MySQL Service URI
- `SECRET_KEY` = long random secret
- `FRONTEND_URL` = your Vercel URL
- `ADMIN_PASSWORD` = strong initial admin password
- `ADMIN_EMAIL` = desired admin email
- `COOKIE_SECURE` = `true`

The three required model files are stored in `backend/models/` and are loaded locally by the Render backend. The backend loads only one model at a time to reduce memory pressure.

After deployment, verify:

`https://YOUR-RENDER-SERVICE.onrender.com/api/health`

The response should report `"status":"ok"` and `"database":"connected"`.

## 3. Vercel frontend

Import this repository into Vercel.

Set the project root directory to:

`frontend`

Vercel should detect Vite automatically.

Add:

`VITE_API_URL=https://YOUR-RENDER-SERVICE.onrender.com`

Deploy.

The React frontend calls the Flask API using credentials-enabled requests. The Render backend only allows the Vercel origin configured through `FRONTEND_URL`.

## 4. Admin account

Do not put the admin password in GitHub.

Set `ADMIN_EMAIL` and `ADMIN_PASSWORD` in Render environment variables. On the first startup, the backend creates the admin account if that email does not already exist.

## 5. Architecture

```
Browser
   |
   v
Vercel / React
   |
   | HTTPS + CORS + session cookie
   v
Render / Flask API
   |
   +---- PyTorch MobileNet
   +---- PyTorch U-Net + ResNet34
   |
   +---- Aiven MySQL
```

The browser never connects directly to Aiven.

## 6. Important deployment behavior

Render's default filesystem is ephemeral, so the backend does not use local SQLite for persistent accounts and does not depend on uploaded files remaining after a request.

Free Render services can spin down when idle, so the first request after inactivity can be slower. This is a platform limitation rather than an application error.

# MEC Paper Trading Bot

Bot de paper trading con la estrategia MEC (Directional Change + Head & Shoulders).

## Portfolio
- **Cuenta A — M1 (40€):** leverage 3×, vol sizing
- **Cuenta B — S4 (160€):** leverage dinámico 2×–5× con filtro momentum

## Setup (5 pasos)

### 1. Crear el repo en GitHub
- Crea un repo **público** (necesario para GitHub Pages gratis)
- Sube todos estos archivos

### 2. Activar GitHub Pages
- Settings → Pages → Source: **Deploy from branch**
- Branch: `main` / Folder: `/docs`
- URL del dashboard: `https://TU_USUARIO.github.io/TU_REPO/`

### 3. Dar permisos de escritura al workflow
- Settings → Actions → General → Workflow permissions
- Selecciona **"Read and write permissions"**
- Guarda

### 4. Configurar el trigger externo (latencia <15s)

Ve a [cron.job.de](https://cron.job.de) y crea un job gratuito:
- **URL:** `https://api.github.com/repos/TU_USUARIO/TU_REPO/actions/workflows/bot.yml/dispatches`
- **Método:** POST
- **Headers:**
  ```
  Authorization: token TU_GITHUB_TOKEN
  Accept: application/vnd.github.v3+json
  Content-Type: application/json
  ```
- **Body:** `{"ref":"main"}`
- **Schedule:** cada 1 minuto

Para obtener el GitHub Token:
- GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
- Permisos necesarios: `workflow`

### 5. Primer test manual
- Actions → "MEC Paper Trading Bot" → "Run workflow"
- Comprueba que se crea/actualiza `docs/data.json`
- Abre el dashboard en GitHub Pages

## Estructura de archivos
```
├── bot.py                        # Script principal
├── state.json                    # Estado persistente (trades, equity)
├── requirements.txt
├── .github/workflows/bot.yml     # Workflow de GitHub Actions
└── docs/
    ├── index.html                # Dashboard
    └── data.json                 # Datos publicados (leídos por el HTML)
```

## Escalar a trading real

Cuando quieras pasar a dinero real:
1. Sustituye la función `open_trade()` en `bot.py` por llamadas a la API de Bitget/Binance
2. Añade las API keys como GitHub Secrets (Settings → Secrets → Actions)
3. Mueve el script a Railway/Render para latencia <5s permanente

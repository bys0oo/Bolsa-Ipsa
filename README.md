# Señales IPSA — Dashboard

Dashboard web que muestra las señales COMPRA/VENTA de acciones del IPSA
(metodología Trading Latino) y se actualiza solo, todos los días, además
de poder correrse manualmente cuando quieras desde el celular.

## Cómo se arma (una sola vez)

1. **Crea un repositorio en GitHub** (gratis): entra a github.com, botón
   "New repository", ponle un nombre (ej. `señales-ipsa`), márcalo como
   **público** (necesario para GitHub Pages gratis), y créalo vacío.

2. **Sube estos archivos** al repo, respetando las carpetas:
   ```
   señales-ipsa/
   ├── index.html
   ├── generar_dashboard_data.py
   ├── trading_latino_chile.py        <- el script que ya tienes
   ├── data/
   │   └── senales.json
   └── .github/
       └── workflows/
           └── actualizar.yml
   ```
   La forma más fácil: en la página del repo vacío, usa "uploading an
   existing file" y arrastra todo (puedes arrastrar carpetas completas
   en el navegador).

3. **Activa GitHub Pages**: en el repo, ve a `Settings` → `Pages` (menú
   de la izquierda) → en "Source" elige `Deploy from a branch` → branch
   `main`, carpeta `/ (root)` → Save. GitHub te va a dar una URL tipo
   `https://tu-usuario.github.io/señales-ipsa/` — esa es la que abres
   desde el celular.

4. **Corre la primera actualización manualmente**: en el repo, pestaña
   `Actions` (arriba) → click en "Actualizar señales IPSA" (a la
   izquierda) → botón `Run workflow` → `Run workflow` de nuevo para
   confirmar. Tarda 1-2 minutos. Cuando termine (ícono verde ✓), refresca
   la página del dashboard.

## Cómo se actualiza después

- **Automático**: corre solo todos los días de semana (~17-18h Chile),
  sin que tengas que hacer nada.
- **Manual, cuando quieras**: entra a github.com desde el celular
  (funciona bien en el navegador del celular, no hace falta la app),
  ve a tu repo → `Actions` → `Actualizar señales IPSA` → `Run workflow`.
  En 1-2 minutos el dashboard queda al día.

## Cartera simulada (sin dinero real)

En cada señal de COMPRA o VENTA hay un botón **"+ Simular"**: ingresa un
monto en CLP (ej. 1.000.000) y queda registrado como una posición de
prueba. El dashboard calcula solo, día a día, cuánto habría ganado o
perdido esa posición según el precio actual.

**Importante:** esto se guarda en el navegador de tu celular (`localStorage`),
no en el repositorio ni en ningún servidor. Eso significa:
- Solo lo vas a ver desde el mismo celular/navegador donde lo registraste.
- Si borras datos de navegación de ese navegador, se pierde el historial.
- Si en algún momento quieres que se sincronice entre varios dispositivos,
  se puede armar con Google Sheets como respaldo — avísame si llegas a
  necesitarlo.

## Si algo falla "no se pudo cargar data/senales.json todavía", es
  que el workflow nunca corrió — ve a `Actions` y córrelo manualmente
  una vez (paso 4 arriba).
- Si el workflow sale con una ❌ roja, entra a verlo (click en la
  corrida) para ver el error — probablemente algún ticker cambió de
  nombre en Yahoo Finance, igual que nos pasó antes con
  `ITAUCORP.SN`/`OROBLANCO.SN`.

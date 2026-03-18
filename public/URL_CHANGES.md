# Frontend URL Changes Required

The HTML from the notebook (HTML_PAGE and LOGIN_HTML variables in cell 19)
needs these fetch() URL rewrites before copying to /public/:

## login page  →  public/index.html
Copy the LOGIN_HTML string content to public/index.html, then change:

  /login      →  /api/auth?action=login
  /register   →  /api/auth?action=register

Specifically in the doLogin() and doSignup() JS functions:
  fetch('/login',    ...)   →  fetch('/api/auth?action=login', ...)
  fetch('/register', ...)   →  fetch('/api/auth?action=register', ...)

After login redirect: window.location.href = '/app'  (unchanged — Vercel rewrites handle this)

## app page  →  public/app.html
Copy the HTML_PAGE string content to public/app.html, then change:

  fetch('/analyze', ...)         →  fetch('/api/analyze', ...)
  fetch('/export-csv')           →  fetch('/api/export-csv')
  fetch('/heatmap')              →  fetch('/api/heatmap')
  fetch('/reset-heatmap', ...)   →  fetch('/api/heatmap/reset', ...)
  fetch('/me')                   →  fetch('/api/auth?action=me')
  href="/logout"                 →  href="/api/auth?action=logout"

These are all in the inline <script> block near the bottom of HTML_PAGE.

## Auth check (app.html)
The /app route in Flask redirected to / if not logged in.
In the static version, add this snippet at the top of the <script> in app.html:

  (async () => {
    const r = await fetch('/api/auth?action=me');
    if (!r.ok) window.location.href = '/';
  })();

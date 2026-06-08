/**
 * Servidor HTTP estatico para la pildora formativa de GANs.
 * No requiere instalacion de paquetes: usa solo modulos nativos de Node.js.
 *
 * Uso:
 *   node webapp/server.js
 *
 * La aplicacion queda disponible en http://localhost:3000
 * Para detenerlo: Ctrl + C
 */

const http = require("http");
const fs   = require("fs");
const path = require("path");

const PORT    = process.env.PORT || 3000;
const PUBLIC  = path.join(__dirname, "public");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css":  "text/css",
  ".js":   "application/javascript",
  ".png":  "image/png",
  ".jpg":  "image/jpeg",
  ".svg":  "image/svg+xml",
  ".ico":  "image/x-icon",
};

http.createServer((req, res) => {
  // Servir siempre index.html para cualquier ruta (SPA)
  let filePath = path.join(PUBLIC, req.url === "/" ? "index.html" : req.url);
  const ext    = path.extname(filePath).toLowerCase();

  // Proteccion contra path traversal
  if (!filePath.startsWith(PUBLIC)) {
    res.writeHead(403); res.end("Forbidden"); return;
  }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      // Si no se encuentra el fichero, servir index.html
      fs.readFile(path.join(PUBLIC, "index.html"), (e2, d2) => {
        if (e2) { res.writeHead(500); res.end("Error interno"); return; }
        res.writeHead(200, { "Content-Type": MIME[".html"] });
        res.end(d2);
      });
      return;
    }
    res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream" });
    res.end(data);
  });

}).listen(PORT, () => {
  console.log(`Servidor arrancado en http://localhost:${PORT}`);
  console.log("Para detenerlo pulsa Ctrl + C");
});

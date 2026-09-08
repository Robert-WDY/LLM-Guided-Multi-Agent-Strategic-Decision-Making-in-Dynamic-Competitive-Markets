// Optional template binding. Local market state lives in the Python API's SQLite store.
declare namespace Cloudflare {
  interface Env { DB?: D1Database }
}

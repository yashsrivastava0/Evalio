export default function Home() {
  return (
    <main style={{ padding: "2rem", fontFamily: "system-ui" }}>
      <h1 className="text-4xl font-bold">Evalio AI Engine</h1>
      <p>SQL-first, deterministic AI commerce engine for smartphones &amp; laptops.</p>
      <p style={{ color: "#666", marginTop: "1rem" }}>
        Frontend stub — full UI coming in later sprints.
      </p>
      <div style={{ marginTop: "2rem", padding: "1rem", background: "#f5f5f5", borderRadius: "8px" }}>
        <h2>API Endpoints</h2>
        <ul>
          <li><code>POST /v1/search</code> — Search products</li>
          <li><code>POST /v1/compare</code> — Compare 2–4 products</li>
          <li><code>POST /v1/scrape</code> — Trigger scraper</li>
          <li><code>GET /health</code> — Health check</li>
        </ul>
      </div>
    </main>
  );
}

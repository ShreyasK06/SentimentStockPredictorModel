import { doc, REPO, type Meta } from "../lib/data";
import { SECTIONS } from "./Nav";

export function Footer({ meta }: { meta: Meta | null }) {
  return (
    <footer className="footer">
      <div className="footer-in">
        <div className="foot-cols">
          <div>
            <h2 className="foot-h">Explore</h2>
            <ul>
              {SECTIONS.map((s) => (
                <li key={s.id}>
                  <a href={`#${s.id}`}>{s.label}</a>
                </li>
              ))}
              <li>
                <a href="#/results">Results explorer</a>
              </li>
            </ul>
          </div>
          <div>
            <h2 className="foot-h">Project</h2>
            <ul>
              <li>
                <a href={REPO} target="_blank" rel="noreferrer">
                  GitHub repository<span className="sr-only"> (opens in a new tab)</span>
                </a>
              </li>
              <li>
                <a href={doc("docs/phase8_results.md")} target="_blank" rel="noreferrer">
                  Phase 8 write-up<span className="sr-only"> (opens in a new tab)</span>
                </a>
              </li>
              <li>
                <a href={doc("docs/phase7_tft_results.md")} target="_blank" rel="noreferrer">
                  Phase 7 write-up<span className="sr-only"> (opens in a new tab)</span>
                </a>
              </li>
            </ul>
          </div>
          <div className="foot-note">
            <h2 className="foot-h">Read this</h2>
            <p>
              <strong>Research, not financial advice.</strong> This is an academic-style project. Its models did not beat a trivial baseline, and nothing here should be used to make investment decisions.
            </p>
            {meta && <p className="foot-meta">Numbers exported {meta.exported_at.slice(0, 10)} from the repository reports.</p>}
          </div>
        </div>
        <p className="wordmark" aria-hidden="true">
          Sentiment Stock Predictor
        </p>
      </div>
    </footer>
  );
}

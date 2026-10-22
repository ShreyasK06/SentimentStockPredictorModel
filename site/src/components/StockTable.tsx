import { money, NEWS_TEXT, newsStance, toneText, type Stock } from "../lib/dashboard";
import { arrowFor, Arrow, Delta } from "./StockCard";

/** Every number the cards show, as a plain table (the accessible twin of the card view). */
export function StockTable({ stocks }: { stocks: Stock[] }) {
  return (
    <div className="table-shell">
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Stock table, scrolls sideways on narrow screens">
        <table className="grid">
          <caption className="sr-only">News sentiment and price for the selected stocks</caption>
          <thead>
            <tr>
              <th scope="col">Stock</th>
              <th scope="col">News sentiment</th>
              <th scope="col" className="num">Tone</th>
              <th scope="col" className="num">Headlines</th>
              <th scope="col" className="num">Price</th>
              <th scope="col" className="num">1 day</th>
              <th scope="col" className="num">5 days</th>
            </tr>
          </thead>
          <tbody>
            {stocks.map((s, i) => {
              const n = newsStance(s.news.tone);
              return (
                <tr key={s.symbol} style={{ animationDelay: `${Math.min(i, 12) * 40}ms` }}>
                  <th scope="row">
                    <strong>{s.symbol}</strong> <span className="muted">{s.name}</span>
                  </th>
                  <td>
                    <span className={`badge small-badge ${n}`}>
                      <Arrow dir={arrowFor(n)} />
                      {NEWS_TEXT[n]}
                    </span>
                  </td>
                  <td className="num">{s.news.tone != null ? toneText(s.news.tone) : "n/a"}</td>
                  <td className="num">{s.news.scored}</td>
                  <td className="num">{s.price ? money(s.price.last) : "n/a"}</td>
                  <td className="num">{s.price && <Delta value={s.price.change_1d} />}</td>
                  <td className="num">{s.price && <Delta value={s.price.change_5d} />}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

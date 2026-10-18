interface Props {
  labels: string[] | null;
  matrix: number[][] | null;
  pending?: boolean;
}

/** Row-normalised confusion matrix. Counts and percentages are printed in every cell, so colour is only a reinforcement. */
export function Confusion({ labels, matrix, pending }: Props) {
  const L = labels ?? ["negative", "neutral", "positive"];
  const m = matrix ?? L.map(() => L.map(() => NaN));
  return (
    <div className={`cm-wrap${pending ? " pending" : ""}`}>
      <table className="cm">
        <caption className="sr-only">Confusion matrix. Rows are the true label, columns are the predicted label. Each cell shows the count and the share of its row.</caption>
        <thead>
          <tr>
            <td className="cm-corner">
              <span className="cm-axis">true \ predicted</span>
            </td>
            {L.map((l) => (
              <th key={l} scope="col">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {m.map((row, i) => {
            const tot = row.reduce((a, b) => a + (Number.isNaN(b) ? 0 : b), 0);
            return (
              <tr key={L[i]}>
                <th scope="row">{L[i]}</th>
                {row.map((v, j) => {
                  const share = tot > 0 && !Number.isNaN(v) ? v / tot : 0;
                  const dark = share > 0.5;
                  return (
                    <td
                      key={j}
                      className={`cm-cell${i === j ? " diag" : ""}${dark ? " on" : ""}`}
                      style={{ ["--share" as string]: share.toFixed(3) }}
                    >
                      {Number.isNaN(v) ? (
                        <span className="cm-na">&mdash;</span>
                      ) : (
                        <>
                          <strong>{Math.round(share * 100)}%</strong>
                          <span>{v.toLocaleString("en-US")}</span>
                        </>
                      )}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      {pending && <p className="cm-pending">Run pending. This fills in when the fine-tuning job writes its confusion matrix.</p>}
    </div>
  );
}

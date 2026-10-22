import { useCountUp } from "../lib/motion";

/** A number that eases up to its value. Screen readers get the final value only, never the in-between frames. */
export function CountUp({ value, run = true, format = (n: number) => String(Math.round(n)) }: { value: number; run?: boolean; format?: (n: number) => string }) {
  const shown = useCountUp(value, run);
  return (
    <>
      <span className="sr-only">{format(value)}</span>
      <span aria-hidden="true">{format(shown)}</span>
    </>
  );
}

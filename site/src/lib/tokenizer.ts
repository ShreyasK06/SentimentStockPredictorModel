/**
 * BERT WordPiece tokenizer (uncased), matching the Hugging Face `bert-base-uncased` / DistilBERT tokenizer.
 *
 *   clean text -> space out CJK -> strip accents -> lowercase -> split on whitespace and punctuation -> WordPiece
 *
 * The vocabulary is `public/model/vocab.txt` (one token per line; the line number is the token id).
 * `scripts/export_site_model.py` writes it, and the output was compared id-for-id against the Python tokenizer.
 */
export interface Tokenizer {
  /** Token ids for `[CLS] ... [SEP]`, truncated to `maxLength`. */
  encode(text: string): number[];
}

const MAX_CHARS_PER_WORD = 100;

const isWhitespace = (ch: string) => ch === " " || ch === "\t" || ch === "\n" || ch === "\r" || /\p{Zs}/u.test(ch);
const isControl = (ch: string) => ch !== "\t" && ch !== "\n" && ch !== "\r" && /\p{C}/u.test(ch);

const isPunctuation = (ch: string) => {
  const cp = ch.codePointAt(0)!;
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
  return /\p{P}/u.test(ch);
};

const isCjk = (cp: number) =>
  (cp >= 0x4e00 && cp <= 0x9fff) ||
  (cp >= 0x3400 && cp <= 0x4dbf) ||
  (cp >= 0x20000 && cp <= 0x2a6df) ||
  (cp >= 0x2a700 && cp <= 0x2b73f) ||
  (cp >= 0x2b740 && cp <= 0x2b81f) ||
  (cp >= 0x2b820 && cp <= 0x2ceaf) ||
  (cp >= 0xf900 && cp <= 0xfaff) ||
  (cp >= 0x2f800 && cp <= 0x2fa1f);

/** Drop control characters, turn all whitespace into plain spaces, and put spaces around CJK characters. */
function clean(text: string): string {
  let out = "";
  for (const ch of text) {
    const cp = ch.codePointAt(0)!;
    if (cp === 0 || cp === 0xfffd || isControl(ch)) continue;
    if (isWhitespace(ch)) out += " ";
    else if (isCjk(cp)) out += ` ${ch} `;
    else out += ch;
  }
  return out;
}

const stripAccents = (s: string) => s.normalize("NFD").replace(/\p{Mn}/gu, "");

/** Split a lowercase word on punctuation so each punctuation mark becomes its own piece. */
function splitPunctuation(word: string): string[] {
  const pieces: string[] = [];
  let current = "";
  for (const ch of word) {
    if (isPunctuation(ch)) {
      if (current) pieces.push(current);
      pieces.push(ch);
      current = "";
    } else {
      current += ch;
    }
  }
  if (current) pieces.push(current);
  return pieces;
}

export function createTokenizer(vocabText: string, maxLength = 64): Tokenizer {
  const lines = vocabText.split(/\r?\n/); // tolerate CRLF in case git rewrote the line endings
  if (lines[lines.length - 1] === "") lines.pop();
  const vocab = new Map<string, number>();
  lines.forEach((token, id) => vocab.set(token, id));

  const need = (token: string) => {
    const id = vocab.get(token);
    if (id === undefined) throw new Error(`Vocabulary is missing ${token}`);
    return id;
  };
  const CLS = need("[CLS]");
  const SEP = need("[SEP]");
  const UNK = need("[UNK]");

  function wordPiece(word: string): number[] {
    const chars = Array.from(word);
    if (chars.length > MAX_CHARS_PER_WORD) return [UNK];
    const ids: number[] = [];
    let start = 0;
    while (start < chars.length) {
      let end = chars.length;
      let found: number | undefined;
      while (start < end) {
        const piece = (start > 0 ? "##" : "") + chars.slice(start, end).join("");
        const id = vocab.get(piece);
        if (id !== undefined) {
          found = id;
          break;
        }
        end--;
      }
      if (found === undefined) return [UNK]; // one unmatched piece makes the whole word unknown
      ids.push(found);
      start = end;
    }
    return ids;
  }

  return {
    encode(text: string): number[] {
      const body: number[] = [];
      for (const raw of clean(text).split(" ")) {
        if (!raw) continue;
        const word = stripAccents(raw).toLowerCase();
        for (const piece of splitPunctuation(word)) body.push(...wordPiece(piece));
      }
      return [CLS, ...body.slice(0, Math.max(0, maxLength - 2)), SEP];
    },
  };
}

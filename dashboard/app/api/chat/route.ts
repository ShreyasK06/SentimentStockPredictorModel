import OpenAI from "openai";
import { createClient } from "@/lib/supabase/server";

// Calls out to NVIDIA's OpenAI-compatible endpoint and streams tokens back
// as the request is in flight — needs the Node.js runtime (not Edge).
export const runtime = "nodejs";

const nvidia = new OpenAI({
  apiKey: process.env.NVIDIA_API_KEY,
  baseURL: "https://integrate.api.nvidia.com/v1",
});

const MODEL = "nvidia/nemotron-3-ultra-550b-a55b";

type ChatRole = "user" | "assistant";

interface IncomingMessage {
  role: ChatRole;
  content: string;
}

interface PageContext {
  pathname?: string;
  pageDescription?: string;
  // Anything beyond these two static fields is live per-page data a client
  // component registered via useSetChatContext (see lib/chat/ChatContext.tsx)
  // — e.g. {strategy, confidenceThreshold, predictions} on /predictions, or
  // {ticker, quote, metrics, dcfInputs} on a stock detail page. Typed loosely
  // here since the shape varies per page; buildSystemPrompt only checks
  // whether any such extra keys are present, and the raw JSON is handed to
  // the model verbatim so it can read the actual values.
  [key: string]: unknown;
}

const STATIC_PAGE_CONTEXT_KEYS = new Set(["pathname", "pageDescription"]);

function hasLivePageData(pageContext?: PageContext): boolean {
  if (!pageContext) return false;
  return Object.keys(pageContext).some(
    (key) => !STATIC_PAGE_CONTEXT_KEYS.has(key) && pageContext[key] !== undefined && pageContext[key] !== null
  );
}

interface ChatRequestBody {
  messages: IncomingMessage[];
  pageContext?: PageContext;
}

// Project facts the assistant is allowed to state as general knowledge —
// mirrors the language on the About page verbatim in substance, so the
// assistant never contradicts what the app itself already tells the user.
const PROJECT_FACTS = `Project facts (general knowledge, not live data):
- This project tested whether sentiment and technical signals could predict short-horizon stock direction across a 28-stock large-cap universe spanning six sectors.
- Five research phases trained and evaluated five algorithms: logistic regression, LightGBM, a random forest, XGBoost, and a small LSTM, against walk-forward out-of-sample splits.
- The finding, stated plainly because it's the load-bearing fact of the whole project: no configuration of any algorithm produced a validated edge over the simplest baseline (always guessing the stock goes up). Confidence gating, cross-sectional features, sector-relative z-scores, alternate feature sets — none of it survived honest evaluation.
- Midway through the project the team found a real bug in their own evaluation code: confidence-filtered accuracy was being compared against the wrong baseline, which made some early results look better than they were. The bug was fixed and every earlier comparison was re-run under the corrected methodology.
- The paper-trading system runs three named strategies: direction-daily is the real one (LightGBM, 10-day horizon, confidence-gated at 0.6, trades a 28-ticker universe with virtual capital). volatility and post-event are both placeholders that currently reuse the direction model's output — no real classifier has been built for either yet, and their signals are logged only, not traded on (except direction-daily).
- The dashboard has these pages: Predictions (direction-daily and post-event predictions), Volatility (the volatility placeholder strategy's logged signals), Paper Trading (the real paper-trading ledger and equity curve), Stocks (watchlist and a DCF intrinsic-value calculator), About (the project narrative).`;

function buildSystemPrompt(pageContext?: PageContext): string {
  let pageLine: string;
  if (!pageContext?.pathname) {
    pageLine = `You don't know what page the user is currently on. Don't guess.`;
  } else if (hasLivePageData(pageContext)) {
    const liveData = { ...pageContext };
    delete liveData.pathname;
    delete liveData.pageDescription;
    pageLine = `The user is currently viewing: ${pageContext.pathname}${
      pageContext.pageDescription ? ` — ${pageContext.pageDescription}` : ""
    }. You DO have a live snapshot of what's actually rendered on this page right now, given below as JSON — use it to answer specifically (exact numbers, tickers, statuses) rather than speaking generally. Only state numbers that appear in this snapshot or that the user typed themselves; if something isn't in it, say you don't have that particular value rather than guessing.

Live page snapshot (JSON):
${JSON.stringify(liveData)}`;
  } else {
    pageLine = `The user is currently viewing: ${pageContext.pathname}${
      pageContext.pageDescription ? ` — ${pageContext.pageDescription}` : ""
    }. You know what this page is *about*, but you do NOT have live access to the specific numbers, tickers, or values currently rendered there. Never pretend otherwise — if the user asks about a specific number on their screen, say plainly that you don't have live access to it, and offer to explain generally how that page/number works instead.`;
  }

  return `You are Scout, the built-in assistant for "Stock Predictor," a research dashboard for a project that tested whether sentiment/technical signals predict stock direction. If asked your name, say you're Scout. You answer questions about the user's own data as it appears in this app (their predictions, watchlist, DCF calculator inputs, paper-trading ledger) and about how this project works. You are a research/data assistant, not a financial advisor.

${pageLine}

Hard rules, non-negotiable:
1. Never give buy/sell recommendations, price targets, or any "should I..." investment advice, for any ticker, ever. If asked, decline plainly and redirect to what you CAN do: explain the data, explain how a number was computed, explain the DCF assumptions, or explain how the project's models/strategies work.
2. Never fabricate a number. Only reference numbers that appear in the live page snapshot above (when present) or that the user explicitly shares with you in the conversation. On pages without a live snapshot, you do not have live access to whatever is currently rendered on their screen — say so honestly rather than inventing or guessing a value.
3. When discussing confidence, a prediction, or a ticker's outlook specifically, naturally note that no strategy in this app has shown a validated statistical edge — weave it in when it's relevant to the question, not as a canned disclaimer on every unrelated message (e.g. don't say it in response to "hi").
4. Never claim to see live data you don't have, and never claim a live snapshot is stale or wrong without evidence. It is always better to admit a limitation than to sound more capable than you are.

${PROJECT_FACTS}

Respond concisely, in plain text, no markdown headers.`;
}

const FALLBACK_ERROR_MESSAGE =
  "Something went wrong reaching the assistant — try again.";

function sseEvent(data: unknown): string {
  return `data: ${JSON.stringify(data)}\n\n`;
}

export async function POST(request: Request) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  let body: ChatRequestBody;
  try {
    body = await request.json();
  } catch {
    return new Response(JSON.stringify({ error: "Invalid request body" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const incoming = Array.isArray(body.messages) ? body.messages : [];
  const messages = incoming
    .filter(
      (m): m is IncomingMessage =>
        !!m &&
        (m.role === "user" || m.role === "assistant") &&
        typeof m.content === "string" &&
        m.content.trim().length > 0
    )
    .slice(-20) // cap conversation history sent upstream
    .map((m) => ({ role: m.role, content: m.content }));

  if (messages.length === 0) {
    return new Response(JSON.stringify({ error: "No message content" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const systemPrompt = buildSystemPrompt(body.pageContext);

  // Establish the upstream stream first so a connection/auth failure with
  // NVIDIA surfaces as a clean error response before we commit to a 200 and
  // start streaming to the client.
  let upstream: Awaited<ReturnType<typeof nvidia.chat.completions.create>>;
  try {
    upstream = await nvidia.chat.completions.create({
      model: MODEL,
      messages: [{ role: "system", content: systemPrompt }, ...messages],
      temperature: 1,
      top_p: 0.95,
      max_tokens: 16384,
      stream: true,
      // NVIDIA-specific, non-standard fields — the openai SDK serializes any
      // extra keys placed directly on the params object into the JSON body
      // (see node_modules/openai/resources/chat/completions/completions.js:
      // `create(body, options) { return this._client.post(path, { body,
      // ...options }) }`, where `body` is exactly the params object passed
      // here). Passing these via the second `options` argument's `body` key
      // instead would *replace* the whole request body rather than merge —
      // confirmed live against the real endpoint, where it dropped `model`
      // and returned a 400. Cast is needed because these fields aren't in
      // the SDK's typed params.
      ...({
        chat_template_kwargs: { enable_thinking: true },
        reasoning_budget: 16384,
      } as Record<string, unknown>),
    });
  } catch (err) {
    console.error("NVIDIA chat completion request failed:", err);
    return new Response(JSON.stringify({ error: FALLBACK_ERROR_MESSAGE }), {
      status: 502,
      headers: { "Content-Type": "application/json" },
    });
  }

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for await (const chunk of upstream) {
          const delta = chunk.choices?.[0]?.delta as
            | { content?: string | null; reasoning_content?: string | null }
            | undefined;
          // Deliberately drop reasoning_content — that's the model's
          // internal chain-of-thought, not a user-facing answer. Only
          // forward actual answer tokens.
          if (delta?.content) {
            controller.enqueue(
              encoder.encode(sseEvent({ type: "token", content: delta.content }))
            );
          }
        }
        controller.enqueue(encoder.encode(sseEvent({ type: "done" })));
      } catch (err) {
        console.error("NVIDIA stream error mid-response:", err);
        controller.enqueue(
          encoder.encode(sseEvent({ type: "error", message: FALLBACK_ERROR_MESSAGE }))
        );
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}

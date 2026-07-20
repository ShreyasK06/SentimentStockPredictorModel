"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { Reveal } from "@/components/Reveal";

export default function LoginPage() {
  const router = useRouter();
  const supabase = createClient();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [magicLinkMode, setMagicLinkMode] = useState(false);
  const [magicLinkSent, setMagicLinkSent] = useState(false);

  async function handlePasswordSignIn(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);

    const { error } = await supabase.auth.signInWithPassword({ email, password });

    setLoading(false);
    if (error) {
      setError(error.message);
      return;
    }
    router.push("/predictions");
    router.refresh();
  }

  async function handleMagicLink(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);

    const { error } = await supabase.auth.signInWithOtp({
      email,
      options: {
        emailRedirectTo: `${window.location.origin}/auth/confirm`,
      },
    });

    setLoading(false);
    if (error) {
      setError(error.message);
      return;
    }
    setMagicLinkSent(true);
  }

  return (
    <div className="flex flex-1 items-center justify-center bg-bg px-4 font-mono text-ink">
      <Reveal
        index={0}
        className="w-full max-w-sm rounded-[var(--radius-lg)] border border-rule bg-surface p-8 shadow-[var(--shadow-card)]"
      >
        <div className="mb-6">
          <p className="text-xs uppercase tracking-widest text-ink-muted">
            Stock Predictor
          </p>
          <h1 className="mt-1 text-lg font-semibold">Sign in</h1>
        </div>

        {!magicLinkMode ? (
          <form onSubmit={handlePasswordSignIn} className="flex flex-col gap-4">
            <label className="flex flex-col gap-1 text-sm">
              <span className="text-ink-secondary">Email</span>
              <input
                type="email"
                required
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="rounded-[var(--radius-md)] border border-rule bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
              />
            </label>
            <label className="flex flex-col gap-1 text-sm">
              <span className="text-ink-secondary">Password</span>
              <input
                type="password"
                required
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="rounded-[var(--radius-md)] border border-rule bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
              />
            </label>

            {error && (
              <p className="rounded-[var(--radius-md)] border border-status-serious bg-status-serious-soft px-3 py-2 text-xs text-status-serious">
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={loading}
              className="mt-2 rounded-[var(--radius-md)] bg-accent px-3 py-2 text-sm font-medium text-white transition-opacity hover:opacity-90 disabled:opacity-50"
            >
              {loading ? "Signing in…" : "Sign in"}
            </button>

            <button
              type="button"
              onClick={() => {
                setError(null);
                setMagicLinkMode(true);
              }}
              className="text-left text-xs text-ink-muted underline underline-offset-2 hover:text-ink-secondary"
            >
              Sign in with a magic link instead
            </button>
          </form>
        ) : (
          <form onSubmit={handleMagicLink} className="flex flex-col gap-4">
            <label className="flex flex-col gap-1 text-sm">
              <span className="text-ink-secondary">Email</span>
              <input
                type="email"
                required
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="rounded-[var(--radius-md)] border border-rule bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
              />
            </label>

            {error && (
              <p className="rounded-[var(--radius-md)] border border-status-serious bg-status-serious-soft px-3 py-2 text-xs text-status-serious">
                {error}
              </p>
            )}

            {magicLinkSent && (
              <p className="rounded-[var(--radius-md)] border border-status-good bg-status-good-soft px-3 py-2 text-xs text-status-good">
                Check your email for a sign-in link.
              </p>
            )}

            <button
              type="submit"
              disabled={loading}
              className="mt-2 rounded-[var(--radius-md)] bg-accent px-3 py-2 text-sm font-medium text-white transition-opacity hover:opacity-90 disabled:opacity-50"
            >
              {loading ? "Sending…" : "Send magic link"}
            </button>

            <button
              type="button"
              onClick={() => {
                setError(null);
                setMagicLinkSent(false);
                setMagicLinkMode(false);
              }}
              className="text-left text-xs text-ink-muted underline underline-offset-2 hover:text-ink-secondary"
            >
              Back to password sign-in
            </button>
          </form>
        )}
      </Reveal>
    </div>
  );
}

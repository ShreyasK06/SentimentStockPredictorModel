import { createBrowserClient } from "@supabase/ssr";

// Browser-side Supabase client for use in Client Components (login form,
// sign-out button). Reads the public env vars from .env.local.
export function createClient() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  );
}

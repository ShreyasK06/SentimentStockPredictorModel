"use client";

import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";

export function SignOutButton() {
  const router = useRouter();
  const supabase = createClient();

  async function handleSignOut() {
    await supabase.auth.signOut();
    router.push("/login");
    router.refresh();
  }

  return (
    <button
      onClick={handleSignOut}
      className="rounded-[var(--radius-md)] border border-rule px-3 py-1.5 text-xs text-ink-secondary transition-colors hover:border-accent hover:text-accent"
    >
      Sign out
    </button>
  );
}

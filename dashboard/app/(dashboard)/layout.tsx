import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { TabNav } from "@/components/TabNav";
import { SignOutButton } from "@/components/SignOutButton";
import { TickerTape } from "@/components/TickerTape";
import { ThemeToggle } from "@/components/ThemeToggle";
import { ChatWidget } from "@/components/ChatWidget";
import { ChatContextProvider } from "@/lib/chat/ChatContext";

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect("/login");
  }

  return (
    <ChatContextProvider>
      <div className="flex min-h-full flex-1 flex-col bg-bg font-mono text-ink">
        <header className="border-b border-rule bg-surface">
          <div className="mx-auto flex max-w-6xl items-center justify-between px-4">
            <div className="flex items-center gap-6">
              <span className="py-2 text-xs uppercase tracking-widest text-ink-muted">
                Stock Predictor
              </span>
              <TabNav />
            </div>
            <div className="flex items-center gap-3">
              <span className="hidden text-xs text-ink-muted sm:inline">{user.email}</span>
              <SignOutButton />
              <ThemeToggle />
            </div>
          </div>
          <div className="border-t border-rule bg-bg">
            <TickerTape />
          </div>
        </header>
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6">{children}</main>
        <ChatWidget />
      </div>
    </ChatContextProvider>
  );
}

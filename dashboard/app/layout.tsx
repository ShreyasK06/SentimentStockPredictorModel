import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Stock Predictor",
  description:
    "Paper-trading dashboard for the sentiment stock predictor research project.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full flex flex-col antialiased">
        {/* Runs before hydration/first paint to avoid a flash of the wrong
            theme: mirrors the "theme" key ThemeToggle.tsx writes to
            localStorage onto <html data-theme>, which globals.css uses to
            override the prefers-color-scheme media query. Defensive
            try/catch since localStorage can throw (private mode, blocked
            storage, etc.) — falls back to the media query in that case. */}
        <script
          dangerouslySetInnerHTML={{
            __html:
              "try{var t=localStorage.getItem('theme');if(t==='light'||t==='dark'){document.documentElement.setAttribute('data-theme',t);}}catch(e){}",
          }}
        />
        {children}
      </body>
    </html>
  );
}

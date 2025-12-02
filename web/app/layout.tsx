import type { Metadata } from "next";
import "./globals.css";
import { EmbeddingModelProvider } from "@/contexts/EmbeddingModelContext";

export const metadata: Metadata = {
  title: "Torah Source Finder",
  description: "Search and manage Torah sources from Sefaria",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="he" dir="rtl">
      <body>
        <EmbeddingModelProvider>{children}</EmbeddingModelProvider>
      </body>
    </html>
  );
}

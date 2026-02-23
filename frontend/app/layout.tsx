import type { Metadata } from "next";

export const metadata: Metadata = {
    title: "Evalio AI Commerce",
    description: "Deterministic AI shopping engine for smartphones & laptops",
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en">
            <body>{children}</body>
        </html>
    );
}

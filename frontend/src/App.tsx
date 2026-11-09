import { useEffect, useState } from "react";

type Health = {
  status: string;
  database: boolean;
  pgvector: string | null;
};

function describe(health: Health | "unreachable" | null): string {
  if (health === null) return "Checking the backend...";
  if (health === "unreachable") return "The backend can't be reached.";
  if (health.status === "ok") return `Connected to the database (pgvector ${health.pgvector}).`;
  return "The backend is running, but the database isn't ready.";
}

export default function App() {
  const [health, setHealth] = useState<Health | "unreachable" | null>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((response) => response.json() as Promise<Health>)
      .then(setHealth)
      .catch(() => setHealth("unreachable"));
  }, []);

  return (
    <main>
      <h1>Tonight's Pick</h1>
      <p role="status">{describe(health)}</p>
    </main>
  );
}

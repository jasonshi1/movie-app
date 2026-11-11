import { useState } from "react";

import FilmTitle from "./FilmTitle";
import Poster from "./Poster";
import { applyProposal, describeProposal, details } from "./format";
import type { ChatReply, FilmMatch, Preferences, Proposal } from "./types";

type Entry =
  | { kind: "person"; text: string }
  | {
      kind: "claude";
      reply: ChatReply;
      proposalStatus: "open" | "saved" | "declined";
    };

type Props = {
  favourites: FilmMatch[];
  preferences: Preferences;
  onPreferencesChange: (preferences: Preferences) => void;
};

export default function Chat({ favourites, preferences, onPreferencesChange }: Props) {
  const [entries, setEntries] = useState<Entry[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const latestPick = [...entries]
    .reverse()
    .find((entry) => entry.kind === "claude" && entry.reply.pick);

  async function send(message: string, rejectMovieId: number | null = null) {
    setSending(true);
    setError(null);
    setEntries((current) => [...current, { kind: "person", text: message }]);
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_id: conversationId,
          message,
          favourites: favourites.map((film) => film.movie_id),
          preferences,
          reject_movie_id: rejectMovieId,
        }),
      });
      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as { detail?: unknown } | null;
        throw new Error(
          typeof body?.detail === "string" ? body.detail : "Something went wrong. Please try again.",
        );
      }
      const reply = (await response.json()) as ChatReply;
      setConversationId(reply.conversation_id);
      setEntries((current) => [...current, { kind: "claude", reply, proposalStatus: "open" }]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Something went wrong.");
    } finally {
      setSending(false);
    }
  }

  function submit(event: React.FormEvent) {
    event.preventDefault();
    const message = draft.trim();
    if (!message) return;
    setDraft("");
    void send(message);
  }

  function rejectLatest() {
    if (!latestPick || latestPick.kind !== "claude" || !latestPick.reply.pick) return;
    const reason = draft.trim();
    setDraft("");
    void send(reason || "Something else, please.", latestPick.reply.pick.film.movie_id);
  }

  function answerProposal(index: number, proposal: Proposal, accept: boolean) {
    if (accept) onPreferencesChange(applyProposal(preferences, proposal));
    setEntries((current) =>
      current.map((entry, position) =>
        position === index && entry.kind === "claude"
          ? { ...entry, proposalStatus: accept ? "saved" : "declined" }
          : entry,
      ),
    );
  }

  function startOver() {
    setEntries([]);
    setConversationId(null);
    setError(null);
  }

  function stopAvoiding(genre: string) {
    onPreferencesChange({
      ...preferences,
      avoid_genres: preferences.avoid_genres.filter((avoided) => avoided !== genre),
    });
  }

  const hasPreferences = preferences.avoid_genres.length > 0 || preferences.max_runtime !== null;

  return (
    <section className="step chat" aria-label="Chat with Claude">
      <h2 className="step-title">Ask Claude</h2>
      <p className="intro">
        Say what you’re in the mood for, like “something tense but not violent, under two hours”.
      </p>

      {hasPreferences && (
        <div className="saved">
          <p className="caption">Remembered</p>
          <ul className="tags" aria-label="Saved preferences">
            {preferences.avoid_genres.map((genre) => (
              <li key={genre}>
                No {genre}
                <button type="button" aria-label={`Stop avoiding ${genre}`} onClick={() => stopAvoiding(genre)}>
                  ×
                </button>
              </li>
            ))}
            {preferences.max_runtime && (
              <li>
                Up to {preferences.max_runtime} min
                <button
                  type="button"
                  aria-label="Remove the runtime limit"
                  onClick={() => onPreferencesChange({ ...preferences, max_runtime: null })}
                >
                  ×
                </button>
              </li>
            )}
          </ul>
        </div>
      )}

      {entries.length > 0 && (
        <ol className="transcript">
          {entries.map((entry, index) =>
            entry.kind === "person" ? (
              <li key={index} className="person">
                <p>{entry.text}</p>
              </li>
            ) : (
              <li key={index} className="claude">
                {entry.reply.pick && (
                  <article className="pick" aria-label="Claude's pick">
                    <Poster url={entry.reply.pick.film.poster_url} className="poster" />
                    <h3>
                      <FilmTitle film={entry.reply.pick.film} />
                    </h3>
                    <p className="details">{details(entry.reply.pick.film)}</p>
                    <p className="reason">{entry.reply.pick.reason}</p>
                    <div className="actions">
                      <a href={entry.reply.pick.film.where_to_watch_url} target="_blank" rel="noreferrer">
                        Where to watch
                      </a>
                      {entry.reply.pick.film.wikipedia_url && (
                        <a href={entry.reply.pick.film.wikipedia_url} target="_blank" rel="noreferrer">
                          Wikipedia
                        </a>
                      )}
                    </div>
                  </article>
                )}
                {entry.reply.reply && <p>{entry.reply.reply}</p>}
                {entry.reply.proposal && entry.proposalStatus === "open" && (
                  <div className="proposal" role="group" aria-label="Suggested preference">
                    <p>{describeProposal(entry.reply.proposal)}?</p>
                    <button type="button" onClick={() => answerProposal(index, entry.reply.proposal!, true)}>
                      Save it
                    </button>
                    <button type="button" onClick={() => answerProposal(index, entry.reply.proposal!, false)}>
                      No thanks
                    </button>
                  </div>
                )}
                {entry.reply.proposal && entry.proposalStatus === "saved" && (
                  <p className="details">Saved: {describeProposal(entry.reply.proposal)}.</p>
                )}
              </li>
            ),
          )}
        </ol>
      )}

      {error && (
        <p className="notice" role="alert">
          {error}
        </p>
      )}

      <form onSubmit={submit}>
        <label htmlFor="chat-message">Your message</label>
        <textarea
          id="chat-message"
          rows={2}
          maxLength={500}
          value={draft}
          placeholder={latestPick ? "Say why, then press Not that one, or ask for something else" : ""}
          onChange={(event) => setDraft(event.target.value)}
        />
        <div className="actions">
          <button className="primary" type="submit" disabled={sending || favourites.length === 0 || !draft.trim()}>
            {sending ? "Thinking…" : "Send"}
          </button>
          {latestPick && (
            <button type="button" disabled={sending} onClick={rejectLatest}>
              Not that one
            </button>
          )}
          {entries.length > 0 && (
            <button type="button" disabled={sending} onClick={startOver}>
              Start over
            </button>
          )}
        </div>
      </form>
      {favourites.length === 0 && <p className="notice quiet">Add at least one favourite first.</p>}
    </section>
  );
}

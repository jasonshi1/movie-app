export type FilmMatch = {
  movie_id: number;
  title: string;
  year: number | null;
  directors: string[];
};

export type PickedFilm = {
  movie_id: number;
  title: string;
  year: number | null;
  genres: string[];
  directors: string[];
  runtime: number | null;
  certification: string | null;
  wikipedia_url: string | null;
  where_to_watch_url: string;
};

export type Preferences = {
  avoid_genres: string[];
  max_runtime: number | null;
};

export type Proposal = {
  kind: "avoid_genre" | "max_runtime";
  genre: string | null;
  minutes: number | null;
  reason: string;
};

export type ChatReply = {
  conversation_id: string;
  reply: string;
  pick: { film: PickedFilm; reason: string } | null;
  proposal: Proposal | null;
};

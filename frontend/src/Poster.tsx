import { useState } from "react";

type Props = {
  url: string | null | undefined;
  className: string;
  // Keep the space with an empty frame when there's no poster, so a list stays aligned.
  keepSpace?: boolean;
};

// A film's poster, the lead image of its Wikipedia article. The title is always next to it, so
// screen readers skip it. If there's none, or it doesn't load, nothing is shown.
export default function Poster({ url, className, keepSpace = false }: Props) {
  const [failed, setFailed] = useState(false);
  if (!url || failed) {
    return keepSpace ? <span className={`${className} empty`} aria-hidden="true" /> : null;
  }
  return (
    <img
      className={className}
      src={url}
      alt=""
      loading="lazy"
      decoding="async"
      referrerPolicy="no-referrer"
      onError={() => setFailed(true)}
    />
  );
}

// A film's title with its year set lighter. Reads as "Seven (1995)", the same as withYear: the
// space stays outside the span, because accessible names trim each element's text.
export default function FilmTitle({ film }: { film: { title: string; year: number | null } }) {
  return (
    <>
      {film.title}
      {film.year && (
        <>
          {" "}
          <span className="year">{`(${film.year})`}</span>
        </>
      )}
    </>
  );
}

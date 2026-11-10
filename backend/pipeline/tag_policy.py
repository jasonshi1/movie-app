"""Decide which Tag Genome tags may appear in an explanation shown to a person.

Every tag still informs ranking. These are only kept out of the visible text.
"""

_SPOILERS = {
    "alternate endings", "bad ending", "great ending", "happy ending", "mindfuck",
    "non-hollywood ending", "plot twist", "powerful ending", "surprise ending", "tricky",
    "twist", "twist ending", "twists & turns", "unusual plot structure", "watch the credits",
}

_NEGATIVE = {
    "bad", "bad acting", "bad cgi", "bad plot", "bad science", "bad script", "bad sequel",
    "book was better", "boring", "boring!", "bullshit history", "cheesy", "complicated plot",
    "confusing", "corny", "crappy sequel", "disappointing", "dumb", "dumb but funny",
    "easily confused with other movie(s) (title)", "horrible", "idiotic", "lame", "no plot",
    "not as good as the first", "not funny", "overrated", "plot holes", "pointless",
    "predictable", "pretentious", "ridiculous", "sappy", "shallow", "slow", "so bad it's funny",
    "so bad it's good", "stupid", "stupid as hell", "stupidity", "too long", "unfunny",
    "unintentionally funny", "unrealistic", "waste of time",
}

_SENSITIVE = {
    "abortion", "aids", "anti-semitism", "autism", "blaxploitation", "cancer", "child abuse",
    "christian", "christianity", "cross dressing", "disability", "erotic", "foul language", "gay",
    "gay character", "genocide", "glbt", "gore", "goretastic", "gory", "gratuitous violence",
    "gross-out", "gruesome", "heroine in tight suit", "holocaust", "homophobia", "homosexuality",
    "incest", "interracial romance", "islam", "jewish", "jews", "judaism", "lesbian", "male nudity",
    "mental illness", "neo-nazis", "notable nudity", "pornography", "prejudice", "prostitution",
    "queer", "race issues", "racism", "rape", "schizophrenia", "segregation", "sex", "sex comedy",
    "sexual", "sexual abuse", "sexuality", "sexualized violence", "sexy", "skinhead", "slavery",
    "strippers", "suicide", "suicide attempt", "terminal illness", "torture", "transgender",
    "voyeurism",
}

_VAGUE_OR_META = {
    "3d", "70mm", "afi 100", "afi 100 (movie quotes)", "awesome", "based on a book",
    "based on book", "better than expected", "better than the american version", "big budget",
    "brilliant", "cgi", "classic", "cool", "criterion", "dialogue", "directorial debut", "effects",
    "enormously long battle scene", "entertaining", "excellent", "exceptional acting",
    "excellent script", "film noir", "fun", "fun movie", "funniest movies", "good", "good acting",
    "good action", "good dialogue", "good music", "good romantic comedies", "good sequel",
    "good soundtrack", "good story", "great", "great acting", "great cinematography",
    "great dialogue", "great movie", "great music", "great soundtrack", "imdb top 250",
    "interesting", "life", "long", "low budget", "made for tv", "masterpiece", "movielens top pick",
    "notable soundtrack", "original", "pg", "pg-13", "perfect", "plot", "prequel", "remade",
    "remake", "saturn award (best special effects)", "script", "sequel", "sequels", "series",
    "special", "story", "suprisingly clever", "time", "trilogy", "underrated", "unique",
    "very funny", "very good", "very interesting", "visual", "visuals",
}

# Spelling variants of a tag that stays visible, so an explanation doesn't repeat itself.
_DUPLICATES = {
    "alien", "android(s)/cyborg(s)", "animation", "based on a comic", "based on a true story",
    "based on a video game", "based on comic", "based on true story", "biography", "biopic",
    "cartoon", "coen bros", "comic", "comic book adaption", "comics", "coming of age",
    "computer animation", "computer game", "cute!", "cyborgs", "dancing", "dark humor",
    "disney animated feature", "distopia", "dragon", "dystopic future", "father son relationship",
    "feel good movie", "fairy tales", "fight scenes", "funny as hell", "gangster", "ghosts/afterlife",
    "hillarious", "hit men", "humorous", "james bond", "007 (series)", "literary adaptation",
    "love story", "melancholic", "monster", "nonlinear", "post apocalyptic", "pixar animation",
    "psychology", "robot", "romantic", "satirical", "sci fi", "science fiction", "scifi",
    "scifi cult", "spy", "spying", "stop motion", "super hero", "super-hero", "superheroes",
    "suspense", "teen", "teenager", "teenagers", "teens", "time", "true story", "vampire",
    "vengeance", "video game", "video game adaptation", "video games", "vigilantism", "violence",
    "werewolf", "witch", "wwii", "zombie",
}

_AWARDS = {"golden palm"}

_HIDDEN = _SPOILERS | _NEGATIVE | _SENSITIVE | _VAGUE_OR_META | _DUPLICATES | _AWARDS
_HIDDEN_PREFIXES = ("nudity", "oscar", "author:", "adapted from:")


def is_displayable(tag: str) -> bool:
    return tag not in _HIDDEN and not tag.startswith(_HIDDEN_PREFIXES)

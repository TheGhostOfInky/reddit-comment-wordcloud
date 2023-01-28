import nltk, string, numpy, os
from itertools import chain
from typing import Optional, List, Dict
from html import unescape
from pmaw import PushshiftAPI
from PIL import Image
from wordcloud import WordCloud
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Loads parameters.toml
with open("parameters.toml", "rb") as f:
    params: dict = tomllib.load(f)

# Get and parse all comments
pshift = PushshiftAPI()
# Parse author and limit from parameters file
author: str = params["reddit"]["username"]
limit: Optional[int] = params["reddit"].get("limit", None)

all_comments = list(pshift.search_comments(author=author, limit=limit))

parsed_comments: List[Dict] = [{
    "body": x["body"],
    "karma": x["score"]
} for x in all_comments]

# Gets stopwords for languages in use
full_stopwords = [nltk.corpus.stopwords.words(x)
                    for x in params["stopwords"]["languages"]]

stopwords: set[str] = set(chain(*full_stopwords))

data: dict[str, int] = {}

# Parses words in comments, removing ponctuation and control characters
for comm in parsed_comments:

    text: str = unescape(comm["body"])
    text.translate(str.maketrans("", "", string.punctuation))

    for chr in params["stopwords"].get("symbols", []):
        text = text.replace(chr, " ")

    words: list[str] = text.lower().split()

    for word in words:

        if word in stopwords:
            continue

        if len(word) < 2:
            continue

        data[word] = data.get(word, 0) + comm["karma"]

# Creates mask from image
mask_path: Optional[str] = params["effects"].get("mask", None)
mask = numpy.array(Image.open(mask_path)) if mask_path else None

# Initializes wordcloud with provided parameters and outputs to file
wc = WordCloud(
    font_path = params["effects"].get("font", None),
    background_color = params["effects"].get("bg_color", None),
    height = params["resolution"].get("height", 400),
    width = params["resolution"].get("width", 600),
    mask = mask,
    colormap = params["effects"].get("color_map", None)
)

wc.generate_from_frequencies(data)

os.makedirs("./image", exist_ok = True)

wc.to_file(f"./image/{author}{'_mask' if mask_path else ''}_image.png")

import nltk,string,numpy,os
from psaw import PushshiftAPI
from PIL import Image
from wordcloud import WordCloud

#Parameters definition
AUTHOR = "TheGhostOfInky"  #Reddit username, "u/" ommited
MASK = None                #Absolute path to mask image
RESOLUTION = (1920,1080)   #(width,height)
FONT = None                #Absolute path to font
BG_COLOR = "#141414"       #Hex string
COLOR_MAP = "winter"       #Matplotlib colormap

#Get and parse all comments
pshift = PushshiftAPI()

all_comments = list(pshift.search_comments(author=AUTHOR,limit=None))

parsed_comments = [{
    "body": x[-1]["body"],
    "karma": x[-1]["score"]
} for x in all_comments]

#Gets stopwords for languages in use
stopwords = set(
    nltk.corpus.stopwords.words("english")
    #nltk.corpus.stopwords.words("romanian")
    )

data = {}

#Parses words in comments, removing ponctuation and control characters
for comm in parsed_comments:

    text = comm["body"]
    text.translate(str.maketrans("","",string.punctuation))

    for chr in ["&amp;#x200B","|","?","!","\"",":",".","*","(",")","-","\\","^",",","{","}","[","]","''","&",";","#"]:
        text = text.replace(chr,"")

    words: list[str] = text.lower().split()

    for word in words:

        if word in stopwords:
            continue
        
        if len(word) < 2:
            continue

        data[word] = data.get(word,0) + comm["karma"]

#Creates mask from image
mask = numpy.array(Image.open(MASK)) if MASK else None

#Initializes wordcloud with provided parameters and outputs to file
wc = WordCloud(
    font_path=FONT,
    background_color=BG_COLOR,
    height=RESOLUTION[1],
    width=RESOLUTION[0],
    mask=mask,
    colormap=COLOR_MAP
)

wc.generate_from_frequencies(data)

os.makedirs("./image",exist_ok=True)

wc.to_file(f"./image/{AUTHOR}{'_mask' if MASK else ''}_image.png")
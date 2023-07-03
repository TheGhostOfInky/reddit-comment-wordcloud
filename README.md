# Reddit comment wordcloud maker

# ⚠️Warning⚠️

In 2023 Pushshift's access to the Reddit API was closed and as a result this
script is not functional at the time of writing (2023-07-03), if the API returns
in the future or a spiritual sucessor with compatible parameters pops up this
script should work only a change in dependencies needed.

## How to use

Download/Clone this repository and install all required modules by running
`pip install -r requirements.txt` or whatever the equivalent in your platform
is, this requires git on Python 3.11 due to the latest `wordcloud` package not
supporting 3.11.

Edit the `parameters.toml` according to the accompanying comments and save the
file.

Run `python main.py` and you should see your new wordcloud in the `image`
folder.

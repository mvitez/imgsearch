## Images search miniapp

This miniapp will index all (or some) of your images and allow you to search them with a textual query.

You need python3. A Nvidia GPU is suggested (for speed).

To install:

`pip3 install -r requirements.txt`

To create the index:

`python3 createindex.py <directory with images>`

To search:

`streamlit run search.py`

This should open a webpage in your browser and allow you to search for images.

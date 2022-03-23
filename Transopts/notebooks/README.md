# Notebooks

All the notebooks and scripts in this folder. This folder will have the following structure:
```
<name of optimisation>.ipynb
<name of optimisation>/
  # file in <name of optimisation>
---------------------------
notebooks/
  - gmlp.ipynb  # notebook will call like this: from gmlp.model import GMLPModel
  - gmlp/       # basically a package style folder
    - model.py
    - conv.py
```

## Datasets

Since this is a benchmarking study it requires a common text dataset. This is how you can get some:

```python
import re
from newspaper import Article

#To get text from the webpage(s)
def get_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

urls = [
    "https://towardsdatascience.com/lucy-says-hi-2031-agi-and-the-future-of-a-i-28b1e7b373f6",
    "https://towardsdatascience.com/to-do-great-data-science-embrace-domain-knowledge-167cb83dc050",
    "https://towardsdatascience.com/geometric-foundations-of-deep-learning-94cdd45b451d",
    "https://kitchingroup.cheme.cmu.edu/blog/2013/01/31/Smooth-transitions-between-discontinuous-functions/"
]
text = "\n\n".join([get_text(u) for u in urls])

#This Regex commands strips the texts so only alphabets, numbers and punctuation mark (.) remain
text = re.sub(r"[^a-z0-9\s\.]", "", text.lower())
```

**NOTE**: These links are used only for education purposes and no commercial activity.


## TODO

- [ ] Modify `autoregressive_wrapper.py` to work for super long sequences with Expire Span model.
